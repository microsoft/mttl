import json
import os
import sys
import torch     
import copy
from torch.optim.optimizer import Optimizer     
import wandb          
from collections import defaultdict
from torch import Tensor, nn
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
        self.accumulate_metrics = defaultdict(list)
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
        self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(batch) # pad tokens also have -100
        padding_mask = self.calculate_routing_mask(batch["input_ids"], self.model.task_id_container["routing_infos"])
        setattr(self.model.task_id_container["routing_infos"], "pad_token_mask", padding_mask)

        outputs = self.model.forward(
            input_ids,
            attention_mask=(input_ids != self.pad_token_id).float(),
        )         
        # output_ids = outputs.logits.argmax(-1)
        # calculate loss, could also be done inside of the model
        bs = input_ids.size(0)
        logits = outputs.logits
        vocab_size = logits.size(-1)
        labels = labels.squeeze(-1)              
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
            loss = loss.view((bs,-1))
            # mean only non-zero
            non_zero_loss = (loss != 0).sum(dim=-1)
            non_zero_loss[non_zero_loss == 0] = 1
            loss = loss.sum(dim=-1) / non_zero_loss
        del outputs, shift_logits, shift_labels
        aux_loss = torch.mean(torch.stack(self.model.task_id_container["routing_infos"].aux_loss)) if len(self.model.task_id_container["routing_infos"].aux_loss)>0 else 0
        
        
        # log metrics if training
        for k,v in self.model.task_id_container["routing_infos"].metrics.items():        
                self.accumulate_metrics[k].append(torch.tensor(v).mean())
        
        
        return loss, aux_loss
    
    def calculate_routing_mask(self, x, routing_infos):  
        # self.args.x_routing_option=4  
        padding_mask = None
        bs, seq = x.shape
        gen_mode = 0                       
        if hasattr(routing_infos, "gen_mode"):
            gen_mode = routing_infos.gen_mode
        if self.args.x_routing_option>0:                                           
            padding_mask = routing_infos.pad_token_mask # 1 if the token is not a pad token, so its either instruciton or output
            instruction_mask = torch.ones_like(padding_mask) # 1 if the token is part of instruction or pad token (so outputs are 0s)
            
            if routing_infos.labels is not None:
                instruction_mask = (routing_infos.labels==-100).float() # 1 if the token is part of instruction or pad token (so outputs are 0s)
            if self.args.x_routing_option==1 or (self.args.x_routing_option==2 and gen_mode): # here we only use instruction to decide about the routing
                padding_mask = padding_mask * instruction_mask # 1 if the token is part of instruction
                
            elif self.args.x_routing_option==2: # or self.args.x_routing_option==3: # here we use instruction and part of the output sofar to decide about the routing, routing will be different for each token
                padding_mask = padding_mask * instruction_mask # only the instruction
                # Find the indices of the last occurrence of 1 in tensor A along the last dimension
                last_ones_indices = padding_mask.sum(dim=1).unsqueeze(-1)#.cpu()                
                
                # Expand dimensions of last_ones_indices to match the shape of B
                expanded_indices = last_ones_indices
                expanded_indices = expanded_indices.repeat(1, seq)
                expanded_indices_inverse = seq - expanded_indices
                expanded_indices_inverse-= torch.arange(seq).unsqueeze(0).to(x.device)
                expanded_indices_inverse = torch.max(expanded_indices_inverse, torch.zeros_like(expanded_indices_inverse))
                expanded_indices_inverse = expanded_indices_inverse.flip(1)
                mask = expanded_indices + expanded_indices_inverse
                mask = mask.unsqueeze(-1).repeat(1,1,seq)
                # shape like mask
                ar = torch.arange(seq).to(x.device)
                ar = ar.unsqueeze(0).unsqueeze(0).repeat(bs, seq, 1)
                
                A = torch.zeros(bs, seq, seq).to(mask.device)
                B = torch.ones(bs, seq, seq).to(mask.device)
                padding_mask = torch.where(ar<mask, A,B)
                padding_mask = 1-padding_mask # per token mask, bs x seq x seq
                del mask, ar, A, B, expanded_indices, expanded_indices_inverse, last_ones_indices
            
            elif self.args.x_routing_option==3:
                padding_mask = padding_mask # 0s only for pad tokens
            elif self.args.x_routing_option==4:
                padding_mask = (padding_mask * instruction_mask, padding_mask)
                
                
            if self.args.x_routing_option==0: # per token routing
                assert padding_mask is None
            elif self.args.x_routing_option==1 or (self.args.x_routing_option==2 and gen_mode): # per example routing, only look at instruction
                assert padding_mask is not None
                assert padding_mask.shape[0]==bs 
                assert padding_mask.shape[1]==seq
            elif self.args.x_routing_option==2 and not gen_mode: # per token, but for each token we look at tokens sofar
                assert padding_mask is not None
                assert padding_mask.dim()==3  
                assert padding_mask.shape[0]==bs
                assert padding_mask.shape[1]==seq
                assert padding_mask.shape[2]==seq
            elif self.args.x_routing_option==3: # during training, we look at the whole seuence (including the outputs), but during generation we only look at what is sofar
                assert padding_mask is not None
                assert padding_mask.shape[0]==bs
                assert padding_mask.shape[1]==seq
            elif self.args.x_routing_option==4:    
                assert isinstance(padding_mask, tuple)
            else:
                raise NotImplementedError()
         
        return padding_mask
        
    

    def generate(self, batch, **kwargs):        
        if not hasattr(self.model, "task_id_container"):
            self.model.task_id_container = {}                          
        self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(batch)    
        setattr(self.model.task_id_container["routing_infos"], "gen_mode", 1)
        padding_mask = self.calculate_routing_mask(batch["input_ids"], self.model.task_id_container["routing_infos"])
        setattr(self.model.task_id_container["routing_infos"], "pad_token_mask", padding_mask)
        return self.model.generate(
            inputs=batch["input_ids"],
            **kwargs
        )
    
    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        # if wandb.run is not None:                
            # pl logger logs every 50 steps, we want to log before every update
        for k,v in self.accumulate_metrics.items():
                # log to wandb directly                   
                # wandb.log({f"train/{k}": torch.tensor(v).mean()})
                self.log(f"train/{k}", torch.tensor(v).mean(), on_step=True)
        self.accumulate_metrics = defaultdict(list)
        return super().on_before_optimizer_step(optimizer, optimizer_idx)
    
    def training_step(self, batch, _):   
        loss, aux_loss = self.forward(batch)
        total_loss = loss+aux_loss
        # outputs = self.model.forward(**batch)   
        # loss= outputs.loss
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)    
        self.log("train/aux_loss", aux_loss, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_epoch=True, prog_bar=True)
        

        for plugin in self.loss_plugins.values():   
            plugin_loss = plugin.compute_loss(self.model, batch)
            loss += plugin.factor * plugin_loss
            self.log(
                f"train/{plugin.name}_loss", plugin_loss, on_epoch=True, prog_bar=True
            )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return total_loss

    def validation_step(self, batch, batch_idx, log=True): 
        self.accumulate_metrics = defaultdict(list)
        loss, aux_loss = self.forward(batch, reduction='none')
        total_loss = loss #+aux_loss 
        # outputs = self.model.forward(**batch, reduction='none')
        # loss= outputs.loss      
        mean_loss = total_loss.sum() / loss.size(0)
        if log:
            self.log("val/loss", mean_loss, on_epoch=True, prog_bar=True)  
            self.log("val/aux_loss", aux_loss, on_epoch=True, prog_bar=True)
            for k,v in self.model.task_id_container["routing_infos"].metrics.items(): 
                self.log(f"val/{k}", torch.tensor(v).mean(), on_epoch=True, prog_bar=True)
            # self.log("val/total_loss", total_loss, on_epoch=True, prog_bar=True)
        return loss, batch['task_ids']

    def on_before_backward(self, loss: Tensor) -> None:
        return super().on_before_backward(loss)
    
    def test_step(self, batch, batch_idx):    
        loss, aux_loss = self.forward(batch, reduction='none')
        return loss, batch['task_ids']
    
    def test_epoch_end(self, outputs):           
        losses = torch.cat([out[0] for out in outputs], 0)
        task_ids = torch.cat([out[1] for out in outputs], 0)
        log_name = f"test/loss"
        if hasattr(self.model, "checkpoint_tested"):    
            log_name = f"test/{self.model.checkpoint_tested}/loss"
        # log per task loss and overall loss
        self.log(log_name, losses.mean(), on_epoch=True, prog_bar=True)
        for task_id in torch.unique(task_ids):
            log_name = f"test/loss_{task_id.item()}"
            if hasattr(self.model, "checkpoint_tested"):
                log_name = f"test/{self.model.checkpoint_tested}/loss_{task_id.item()}"
            self.log(
                log_name,        
                losses[task_ids == task_id].mean(),
                on_epoch=True,
                prog_bar=True,
            )
        return losses
    
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
