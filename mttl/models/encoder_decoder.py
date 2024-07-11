import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForSeq2SeqLM

from mttl.models.get_optimizer import get_optimizer
from mttl.models.get_scheduler import get_scheduler
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.routing import RoutingInfo
from mttl.models.utils import EfficientCheckpointModule, get_global_batch_size
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
                self.args.model, cache_dir=self.args.cache_dir
            )

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

        self._inference_outputs = []
        self._inference_references = []
        self.test_results = []
        self.best_val_result = None

    def add_loss_plugin(self, plugin):
        if self.loss_plugins is not None:
            self.loss_plugins[plugin.name] = plugin
        else:
            self.loss_plugins = nn.ModuleDict({plugin.name: plugin})

    def teacher_force_step(self, batch, reduction="mean"):
        input_ids, target_ids = batch["input_ids"], batch["labels"]

        self.model.info_container["routing_infos"] = RoutingInfo.from_batch(batch)

        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(target_ids)
        # need to transform -100 into padding tokens
        decoder_input_ids[decoder_input_ids == -100] = self.tokenizer.pad_token_id

        outputs = self.model.forward(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=batch["attention_mask"],
            decoder_attention_mask=torch.ones_like(decoder_input_ids).float(),
        )
        loss, _ = label_smoothed_nll_loss(
            F.log_softmax(outputs.logits, dim=-1),
            target_ids,
            epsilon=0.1,
            ignore_index=-100,
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
        self._inference_outputs.append((loss, batch["task_ids"]))
        return loss, batch["task_ids"]

    def on_validation_epoch_end(self):
        outputs = self._inference_outputs
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
        self._inference_outputs.clear()

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
        attention_mask = batch["attention_mask"]

        self.model.info_container["routing_infos"] = RoutingInfo.from_batch(batch)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
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

    def inference_end(self, inference_outputs, inference_labels, split):
        import itertools

        from mttl.evaluators.ni_evaluator import compute_metrics, decode

        inference_outputs = list(
            itertools.chain(
                *[decode(outputs, self.tokenizer) for outputs in inference_outputs]
            )
        )
        inference_labels = list(itertools.chain(*inference_labels))
        metrics = compute_metrics(
            inference_outputs, [[r] for r in inference_labels], reduction="mean"
        )

        to_log = {}
        to_log[f"{split}/em_perf"] = metrics["exact_match"]
        to_log[f"{split}/metric_perf"] = metrics["rougeL"]

        self.log_dict(to_log, on_epoch=True, prog_bar=True)

        with open(self.hparams.output_dir + f"/{split}_predictions.json", "w") as f:
            data = [
                {"prediction": p, "reference": r}
                for p, r in zip(inference_outputs, inference_labels)
            ]
            json.dump(data, f)

        metrics = {
            f"{split}/em": metrics["exact_match"],
            f"{split}/metric_perf": metrics["rougeL"],
        }
        metrics["split"] = split
        metrics["epoch"] = self.current_epoch
        metrics["step"] = self.global_step
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
        output = self.inference_step(batch)
        self._inference_references.append(batch["labels_texts"])
        self._inference_outputs.append(output.cpu())
        return output

    def test_step(self, batch, batch_idx):
        output = self.inference_step(batch)
        self._inference_references.append(batch["labels_texts"])
        self._inference_outputs.append(output.cpu())
        return output

    def on_validation_epoch_end(self):
        outputs = self.inference_end(
            self._inference_outputs, self._inference_references, "val"
        )
        self._inference_outputs.clear()
        self._inference_references.clear()
        return outputs

    def on_test_epoch_end(self):
        outputs = self.inference_end(
            self._inference_outputs, self._inference_references, "test"
        )
        self._inference_outputs.clear()
        self._inference_references.clear()
        return outputs

    def on_training_epoch_end(self, losses):
        avg_loss = (sum([x["loss"] for x in losses]) / len(losses)).item()
        lrs = [x["lr"] for x in self.optimizers().param_groups]
        print(f"loss : {avg_loss:.4f}\tlr {lrs}\n")
