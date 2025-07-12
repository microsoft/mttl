import json
import os
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForSeq2SeqLM

from mttl.models.get_optimizer import get_optimizer
from mttl.models.get_scheduler import get_scheduler
from mttl.models.modify_model import modify_transformer
from mttl.models.utils import (
    EfficientCheckpointModule,
    RoutingInfo,
    get_global_batch_size,
)
from mttl.utils import freeze_embeds, label_smoothed_nll_loss


class EncoderDecoder(EfficientCheckpointModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # log hyperparameters
        self.save_hyperparameters(ignore=["tokenizer", "model_object"])
        self.args = self.hparams

        self.tokenizer = kwargs["tokenizer"]
        self.pad_token_id = self.tokenizer.pad_token_id

        if kwargs.get("model_object") is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.model, cache_dir="/tmp/hf-cache"
            )
            # free-up temporary space
            os.system("/bin/rm -rf /tmp/hf-cache")
            os.system("df")

            if "t5" or "T0" in self.args.model:
                self.pad_token_id = self.tokenizer.pad_token_id
                if (
                    hasattr(self.model.config, "max_position_embeddings")
                    and self.model.config.max_position_embeddings
                    < self.args.max_input_length
                ):
                    print(
                        f"Increasing the model's number of position embedding vectors from {self.model.config.max_position_embeddings} "
                        f"to {self.args.max_input_length}."
                    )
                    self.model.resize_position_embeddings(self.args.max_input_length)
            elif "bart" in self.args.model:
                self.pad_token_id = self.model.model.shared.padding_idx
            else:
                raise NotImplementedError()

            self.model = modify_transformer(self.model, self.args)
            print(self.args)

            if self.args.freeze_embeds:
                print("Freezing embeddings")
                freeze_embeds(self.model)
        else:
            self.model = kwargs.get("model_object")
        self.loss_plugins = nn.ModuleDict({})

        self.test_results = []
        self.best_val_result = None

    def add_loss_plugin(self, plugin):
        if self.loss_plugins is not None:
            self.loss_plugins[plugin.name] = plugin
        else:
            self.loss_plugins = nn.ModuleDict({plugin.name: plugin})

    def teacher_force_step(self, batch, reduction="mean"):
        input_ids, target_ids = batch["input_ids"], batch["target_ids"]

        self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(batch)

        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(target_ids)
        outputs = self.model.forward(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=(input_ids != self.pad_token_id).float(),
            decoder_attention_mask=torch.ones_like(decoder_input_ids).float(),
        )
        loss, _ = label_smoothed_nll_loss(
            F.log_softmax(outputs.logits, dim=-1),
            target_ids,
            epsilon=0.1,
            ignore_index=self.pad_token_id,
            reduction=reduction,
        )
        return loss

    def training_step(self, batch, _):
        loss = self.teacher_force_step(batch)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)

        for plugin in self.loss_plugins.values():
            plugin_loss = plugin.compute_loss(self.model, batch)
            loss += plugin.factor * plugin_loss
            self.log(
                f"train/{plugin.name}_loss", plugin_loss, on_epoch=True, prog_bar=True
            )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.teacher_force_step(batch, reduction="none")
        mean_loss = loss.sum() / loss.size(0)
        self.log("val/loss", mean_loss, on_epoch=True, prog_bar=True)
        return loss, batch["task_ids"]

    def validation_epoch_end(self, outputs):
        losses = torch.cat([out[0].sum(-1) for out in outputs], 0)
        task_ids = torch.cat([out[1] for out in outputs], 0)

        # compute the loss per task id
        with open(
            os.path.join(self.args.output_dir, "val_loss_by_task.txt"), "a+"
        ) as f:
            task_losses = {}
            for task_id in torch.unique(task_ids):
                task_losses[task_id.item()] = losses[task_ids == task_id].mean().item()
            f.write(json.dumps(task_losses) + "\n")

    def configure_optimizers(self):
        args = self.args
        self.ml_optimizer = self.ml_scheduler = None

        optimizer, self.trainable_param_names = get_optimizer(
            self, args, no_decay=["bias", "LayerNorm.weight"]
        )
        global_bs = get_global_batch_size(
            args.train_batch_size, args.gradient_accumulation_steps
        )

        if args.total_steps == -1:
            args.total_steps = (
                len(self.trainer.datamodule.train_dataset) // global_bs
            ) * self.trainer.max_epochs

        if args.warmup_steps == -1:
            args.warmup_steps = int(args.warmup_proportion * args.total_steps)

        args.scheduler = "linear_decay_with_warmup"
        scheduler = get_scheduler(optimizer, args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class Finetuner(EncoderDecoder):
    def add_missing_args(self, args):
        for name, value in args.items():
            if name not in self.hparams or self.hparams[name] in ["", None]:
                self.hparams[name] = value
                setattr(self.args, name, value)

    def inference_step(self, batch):
        """used for both validation and testing"""
        input_ids = batch["input_ids"]

        self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(batch)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=(input_ids != self.pad_token_id).float(),
            num_beams=self.args.num_beams,
            max_length=self.args.max_output_length,
            decoder_start_token_id=self.model.config.bos_token_id,
            early_stopping=False,
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

        metrics = data_wrapper.evaluate(predictions, split)

        to_log = {}
        for metric_name, metric in zip(["em", "metric"], metrics):
            to_log[f"{split}/{metric_name}_perf"] = metric

        self.log_dict(to_log, on_epoch=True, prog_bar=True)

        test_examples = data_wrapper.test_examples

        for test_example, prediction in zip(test_examples, predictions):
            test_example["prediction"] = prediction

        with open(self.hparams.output_dir + f"/{split}_predictions.json", "w") as f:
            json.dump(test_examples, f)

        metrics = {f"{split}/em": metrics[0], f"{split}/metric_perf": metrics[1]}
        metrics["split"] = split
        metrics["epoch"] = self.current_epoch
        metrics["step"] = self.global_step
        metrics["metric"] = data_wrapper.metric
        metrics["seed"] = self.hparams.seed

        result_str = json.dumps(metrics) + "\n"
        with open(self.hparams.output_dir + f"/{split}_scores.jsonl", "a+") as f:
            f.write(result_str)

        print(result_str)

        if split == "val":
            if self.best_val_result is None:
                self.best_val_result = metrics
            else:
                if self.best_val_result["val/metric_perf"] < metrics["val/metric_perf"]:
                    self.best_val_result = metrics
        elif split == "test":
            self.test_results.append(metrics)
        return metrics

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def validation_epoch_end(self, outputs):
        return self.inference_end(
            outputs, self.trainer.datamodule.dataset_reader, "val"
        )

    def on_test_epoch_end(self, outputs):
        return self.inference_end(
            outputs, self.trainer.datamodule.dataset_reader, "test"
        )

    def training_epoch_end(self, losses):
        avg_loss = (sum([x["loss"] for x in losses]) / len(losses)).item()
        lrs = [x["lr"] for x in self.optimizers().param_groups]
        print(f"loss : {avg_loss:.4f}\tlr {lrs}\n")
