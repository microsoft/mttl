import json
import math

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from bart import BartWithAdapter
from t5 import T5WithAdapter
from utils import average_dicts, freeze_embeds


class EncoderDecoder(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.args = self.hparams

        if "t5" in self.args.model:
            model = T5WithAdapter.from_pretrained(self.args)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
            self.pad_token_id = self.tokenizer.pad_token_id
            if (
                hasattr(model.config, "max_position_embeddings")
                and model.config.max_position_embeddings < self.args.max_input_length
            ):
                print(
                    f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                    f"to {self.args.max_input_length}."
                )
                model.resize_position_embeddings(self.args.max_input_length)
        elif "bart" in self.args.model:
            model = BartWithAdapter.from_pretrained(self.args)
            self.pad_token_id = model.model.shared.padding_idx
        else:
            raise NotImplementedError()

        self.model = model
        print(self.args)

        if self.args.freeze_embeds:
            print("Freezing embeddings")
            freeze_embeds(model)

    def teacher_force_step(self, batch):
        loss = self.model.skilled_forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            decoder_input_ids=batch["dec_input_ids"],
            decoder_attention_mask=batch["dec_attn_mask"],
            task_ids=batch["task_ids"],
            is_training=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.teacher_force_step(batch)

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.teacher_force_step(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def log_polytropon_metrics(self):
        if self.args.finetune_full_model:
            return

        # look at some stats
        def layer_stats(Z):
            prob = torch.sigmoid(Z)
            discreteness = (
                Bernoulli(logits=Z).entropy().sum().item()
                / np.log(2)
                / np.prod(Z.shape)
            )
            sparsity = (prob + 0.5).floor().mean()
            categorical = prob.mean(0) / prob.mean(0).sum()
            eff = (
                Categorical(probs=categorical).entropy() / math.log(Z.size(-1))
            ).item()

            return {
                "sparsity": sparsity,
                "discreteness_fixed": discreteness,
                "eff_fixed": eff,
            }

        # iterate over encoder and decoder layers
        stats = {"encoder": [], "decoder": []}

        for coder in stats.keys():
            mod = getattr(getattr(self.model, "model", self.model), coder)
            for module in mod.modules():
                if hasattr(module, "module_logits"):
                    Z = module.module_logits[self.trainer.datamodule.training_on]
                    stats[coder] += [layer_stats(Z)]

            # average over layers
            if len(stats[coder]):
                stats[coder] = average_dicts(stats[coder])

                for k, v in stats[coder].items():
                    self.log(f"Z/{coder}.{k}", v, on_epoch=True)

    def on_train_epoch_start(self):
        if self.args.selector not in ["polytropon"]:
            return

        self.log_polytropon_metrics()

    def configure_optimizers(self):
        model = self.model
        args = self.args

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and p.requires_grad
                    and "module_logits" not in n
                ],
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and p.requires_grad
                    and "module_logits" not in n
                ],
                "weight_decay": 0.0,
                "lr": args.learning_rate,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "module_logits" in n
                ],
                "weight_decay": 0.0,
                "lr": args.module_logits_learning_rate,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)

        try:
            global_bs = args.train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
        except:
            global_bs = args.train_batch_size * args.gradient_accumulation_steps

        if args.total_steps == -1:
            args.total_steps = (len(self.trainer.datamodule.train_ds) // global_bs) * self.trainer.max_epochs

        if args.warmup_steps == -1:
            args.warmup_steps = int(args.warmup_proportion * args.total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, ckpt):
        if self.args.finetune_full_model:
            return

        state_keys = [x for x in ckpt["state_dict"].keys()]
        for key in state_keys:
            if (
                all(x not in key for x in ["selector", "combinor", "processor"])
                or "old_linear" in key
            ):
                del ckpt["state_dict"][key]

    def on_load_checkpoint(self, ckpt):
        # ensure that no important key is missing --> note that we are indeed loading twice
        keys = self.load_state_dict(ckpt["state_dict"], strict=False)

        for key in keys.missing_keys:
            assert (
                all(x not in key for x in ["selector", "combinor", "processor"])
                or "old_linear" in key
            )

        assert len(keys.unexpected_keys) == 0


class Finetuner(EncoderDecoder):
    def add_missing_args(self, args):
        for (name, value) in args.items():
            if name not in self.hparams or self.hparams[name] in ["", None]:
                self.hparams[name] = value
                setattr(self.args, name, value)

    def inference_step(self, batch):
        """used for both validation and testing"""
        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            num_beams=self.args.num_beams,
            max_length=self.args.max_output_length,
            decoder_start_token_id=self.model.config.bos_token_id,
            early_stopping=False,
            task_ids=batch["task_ids"],
            task_embeds=batch.get("task_embeds", None),
        )
        return outputs

    def training_step(self, batch, batch_idx):
        loss = self.teacher_force_step(batch)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return loss

    def inference_end(self, inf_step_outputs, data_wrapper, split):
        all_predictions = [
            data_wrapper.decode(out) for output in inf_step_outputs for out in output
        ]
        predictions = all_predictions
        metrics = data_wrapper.evaluate(predictions)

        to_log = {}
        for metric_name, metric in zip(["em", "metric"], metrics):
            to_log[f"{split}/{metric_name}_perf"] = metric

        self.log_dict(to_log, on_epoch=True, prog_bar=True)

        test_examples = data_wrapper.test_examples
        for test_example, prediction in zip(test_examples, predictions):
            test_example["prediction"] = prediction

        with open(self.hparams.output_dir + "/test_predictions.json", "w") as f:
            json.dump(test_examples, f)

        return metrics

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def validation_epoch_end(self, outputs):
        self.inference_end(outputs, self.trainer.datamodule.val_wrapper, "val")

    def test_epoch_end(self, outputs):
        self.inference_end(outputs, self.trainer.datamodule.test_wrapper, "test")

    def training_epoch_end(self, losses):
        avg_loss = (sum([x["loss"] for x in losses]) / len(losses)).item()
        lrs = [x["lr"] for x in self.optimizers().param_groups]
        print(f"loss : {avg_loss:.4f}\tlr {lrs}\n")

    def on_save_checkpoint(self, ckpt):
        # save the whole model, or else `trainer.test` fails
        pass
