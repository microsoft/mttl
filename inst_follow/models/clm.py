import json
import os
import sys
import torch     
import copy       
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

sys.path.append("/home/v-oostapenko/dev/mttl")
from mttl.models.get_optimizer import get_optimizer
from mttl.models.get_scheduler import get_scheduler 
from mttl.models.modify_model import modify_transformer
from mttl.models.utils import EfficientCheckpointModule, RoutingInfo, get_global_batch_size
from mttl.utils import freeze_embeds
from pytorch_lightning.utilities.parsing import AttributeDict

class CLM(EfficientCheckpointModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # log hyperparameters
        self.save_hyperparameters(ignore=["tokenizer", "model_object"])
        self.args = self.hparams

        self.tokenizer = kwargs["tokenizer"]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.model: AutoModelForCausalLM = None

        if kwargs.get('model_object') is None:
            raise NotImplementedError()
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.model, cache_dir=os.environ.get('TRANSFORMERS_CACHE', "/tmp/hf-cache")
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
            self.model = kwargs.get('model_object')
        self.loss_plugins = nn.ModuleDict({})

        self.test_results = []
        self.best_val_result = None

    def add_loss_plugin(self, plugin):
        if self.loss_plugins is not None:
            self.loss_plugins[plugin.name] = plugin
        else:
            self.loss_plugins = nn.ModuleDict({plugin.name: plugin})

    def forward(self, batch, reduction='mean'): 
        input_ids, labels = batch["input_ids"], batch["labels"]       
        self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(batch)
        outputs = self.model.forward(
            input_ids,
            attention_mask=(input_ids != self.pad_token_id).float(),
        )         
        output_ids = outputs.logits.argmax(-1)
        # calculate loss, could also be done inside of the model
        bs = input_ids.size(0)
        logits = outputs.logits
        vocab_size = logits.size(-1)
        labels = labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        #reshape back  
        if reduction == 'none':
            loss = loss.view((bs,-1)).mean(-1)
        return loss

    def generate(self, batch, **kwargs):                                      
        self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(batch)
        return self.model.generate(
            inputs=batch["input_ids"],
            **kwargs
        )
    
    def training_step(self, batch, _):
        loss = self.forward(batch)
        # outputs = self.model.forward(**batch)   
        # loss= outputs.loss
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
        loss = self.forward(batch, reduction='none')
        # outputs = self.model.forward(**batch, reduction='none')
        # loss= outputs.loss
        mean_loss = loss.sum() / loss.size(0)
        self.log("val/loss", mean_loss, on_epoch=True, prog_bar=True)
        return loss, batch['task_ids']

    def validation_epoch_end(self, outputs):
        losses = torch.cat([out[0] for out in outputs], 0)
        task_ids = torch.cat([out[1] for out in outputs], 0)

        # compute the loss per task id
        with open(os.path.join(self.args.output_dir, "val_loss_by_task.txt"), "a+") as f:
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
        global_bs = get_global_batch_size(args.train_batch_size, args.gradient_accumulation_steps)

        if args.total_steps == -1:
            args.total_steps = (
                len(self.trainer.datamodule.train_dataset) // global_bs
            ) * self.trainer.max_epochs

        if args.warmup_steps == -1:
            args.warmup_steps = int(args.warmup_proportion * args.total_steps)

        # args.scheduler = "linear_decay_with_warmup"
        scheduler = get_scheduler(optimizer, args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }    
    
    @property
    def hparams_initial(self): # to make wandb logger work we need to override this method
        """The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only.
        Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`.

        Returns:
            AttributeDict: immutable initial hyperparameters
        """
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        # prevent any change     
        hparams_initial=copy.deepcopy(self._hparams_initial)
        # pop anything that is not json serializable
        hparams_initial.pop('_updated_kwargs')
        return hparams_initial
