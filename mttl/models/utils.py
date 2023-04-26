from pytorch_lightning import LightningModule
import torch
from dataclasses import dataclass
from typing import List

import mttl
from transformers.models.t5.modeling_t5 import T5LayerNorm

class EfficientCheckpointModule(LightningModule):
    """Efficiently save and load checkpoints.
    
    Only saves and loads parameters that are either in the trainable parameters
    or have been loaded from a previous checkpoint.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.save_if_loaded = kwargs.get("save_if_loaded", True)

    def load_state_dict(self, ckpt, **kwargs):
        """Always load state dict with strict = False, so that we can
        early stop successfully.
        """
        # store params that might have been loaded from a previous checkpoint
        self._params_from_checkpoint = (
            set(ckpt.keys()) if self.save_if_loaded else set()
        )
        print("Loading keys...", list(ckpt.keys()))
        return super().load_state_dict(ckpt, strict=False)

    def on_save_checkpoint(self, ckpt):
        if not hasattr(self, "_params_from_checkpoint"):
            self._params_from_checkpoint = set()

        # remove also parameters in the loss plugins, these need not be saved
        # (auxiliary parameters for the losses)
        plugin_param_keys = set()
        for _, plugin in self.loss_plugins.items():
            plugin_param_keys.update(plugin.state_dict().keys())

        keys = [k for k in ckpt["state_dict"].keys()]

        for key in keys:
            # we can safely avoid dumping this parameter if it is both
            # not in the trainable parameters and was not loaded from checkpoint
            if (
                not (key in self.trainable_param_names)
                and not (key in self._params_from_checkpoint)
            ) or key in plugin_param_keys:
                del ckpt["state_dict"][key]
                print("Deleting from state dict:", key)

    def on_load_checkpoint(self, ckpt):
        print("Loading checkpoint...")

        load_result = self.load_state_dict(ckpt["state_dict"])

        assert (
            len(load_result.unexpected_keys) == 0
        ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"


@dataclass
class RoutingInfo:
    task_ids: torch.Tensor
    hashes: List[str]
    instruction_hashes: List[str] = None
    example_ids: List[int] = None

    @classmethod
    def from_batch(cls, batch):
        return RoutingInfo(
            task_ids=batch["task_ids"],
            hashes=batch.get("hashes", None),
            example_ids=batch.get("example_ids", None),
            instruction_hashes=batch.get("instruction_hashes", None)
        )

    def repeat_interleave(self, repeats):
        # useful for beam search
        self.task_ids = self.task_ids.repeat_interleave(repeats)
        if self.hashes:
            self.hashes = [h for h in self.hashes for _ in range(repeats)]
        if self.instruction_hashes:
            self.instruction_hashes = [h for h in self.instruction_hashes for _ in range(repeats)]
        self.example_ids = (
            self.example_ids.repeat_interleave(repeats)
            if self.example_ids is not None
            else None
        )


def get_global_batch_size(batch_size, accumulation_steps):
    """Computes the global batch size."""
    try:
        world_size = torch.distributed.get_world_size()
    except:
        world_size = 1
    global_bs = batch_size * world_size * accumulation_steps
    return global_bs


def set_grad_monitor_hooks(model, config):
    """
    Attach hooks that monitor some gradient alignment stuff
    """

    def fwd_hook(mod, inp, out):
        assert isinstance(inp, tuple) and len(inp) == 1 and isinstance(out, torch.Tensor)
        # We need to keep in memory the output in order to recompute the gradient
        mod.output = out
        mod.input = inp[0]

    def build_bwd_hook(n_tasks):
        def bwd_hook(mod, grad_inp, grad_out):
            assert isinstance(grad_inp, tuple) and len(grad_inp) == 1
            assert isinstance(grad_out, tuple) and len(grad_out) == 1

            task_ids = mod.task_id['routing_infos'].task_ids
            
            # `grad_inp` is the gradient w.r.t to the function's input. 
            # For a linear layer, this would be `bs, seq_dim, d_in`
            # `grad_out` for a linear layer would be  `bs, seq_dim, d_out`

            grad_inp = grad_inp[0]
            grad_out = grad_out[0]
            bs, S, D_in = grad_inp.size()
            bs, S, D_out = grad_out.size()

            if isinstance(mod, mttl.models.poly.PolyLoRALinear) or isinstance(mod, torch.nn.Linear):
                # --- Linear layer specific --- #
                # per example grad of W
                grad_W = torch.einsum('BSO,BSI->BOI', (grad_out.float(), mod.input.float()))
                # grad_X = torch.einsum('BSO,OI->BSI', (grad_out, mod.weight))
                # assert torch.allclose(grad_X, grad_inp)
                per_ex_grad_W = grad_W.view(bs, -1)
                D = per_ex_grad_W.size(-1)
                # --- Linear layer specific --- #
            elif isinstance(mod, T5LayerNorm):
                # --- Layer Norm Specific --- #

                # can we rebuild weight grad ? 
                unmod_hidden_states = mod.output / mod.weight
                per_ex_grad_W = (unmod_hidden_states * grad_out).sum(1)
                bs, D = per_ex_grad_W.size()

                # This line validates that our gradient computation is correct
                # assert torch.allclose(per_ex_grad.sum(0), mod.weight.grad)
                # --- Layer Norm Specific --- #
            else:
                raise TypeError

            # per_ex_grad_W : bs, S, D

            # average across the sequence length
            attn_mask = mod.task_id['enc_attn_mask']
            avg_grad_inp = (grad_inp * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(-1, keepdim=True)
            avg_grad_out = (grad_out * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(-1, keepdim=True)

            grads = {'W' : per_ex_grad_W, 'I': avg_grad_inp, 'O': avg_grad_out}

            if not hasattr(mod, 'grad_container'):
                # build containers
                mod.grad_container = {
                    'W': torch.zeros(n_tasks, D, device=per_ex_grad_W.device), 
                    'I': torch.zeros(n_tasks, D_in, device=per_ex_grad_W.device), 
                    'O': torch.zeros(n_tasks, D_out, device=per_ex_grad_W.device)
                }
                mod.seen_tasks = torch.zeros(n_tasks, device=per_ex_grad_W.device, dtype=torch.bool)

            # accumulate gradients
            unique_tasks = task_ids.unique()
            n_unique_tasks = unique_tasks.size(0)
            task_values, task_index = unique_tasks.sort()
            id_to_index = torch.zeros(n_tasks, dtype=torch.int64, device=grad_inp.device) - 1
            id_to_index[task_values] = task_index

            for name, grad in grads.items():
                container = mod.grad_container[name]
                total_grad = torch.zeros(n_unique_tasks, container.size(-1), device=task_ids.device)
                total_grad.scatter_add_(0, id_to_index[task_ids].view(-1, 1).expand_as(grad), grad)
                total_grad = torch.nn.functional.normalize(total_grad, dim=-1, p=2)

                # new tasks ?
                is_new = mod.seen_tasks[task_values] == False
                if is_new.any():
                    new_tasks = task_values[is_new]
                    index = torch.where(is_new)[0]
                    container[new_tasks] = total_grad[index]
                    mod.seen_tasks[new_tasks] = True
                if (~is_new).any():
                    old_tasks = task_values[~is_new]
                    index = torch.where(~is_new)[0]
                    # gradient EMA
                    container[old_tasks] = torch.nn.functional.normalize(
                        0.99 * container[old_tasks] + 0.01 * total_grad[index]
                    )
        
        return bwd_hook

    # carefully chosen layers to monitor
    last_enc_attn = model.model.encoder.block[-1].layer[0].SelfAttention.o
    last_enc_ffn = model.model.encoder.block[-1].layer[-1].DenseReluDense.wo
    last_enc_norm = model.model.encoder.final_layer_norm
    
    hooks = {
        'last_enc_attn': last_enc_attn,
        'last_enc_ffn' : last_enc_ffn, 
        'last_enc_norm' : last_enc_norm
    }

    for name, layer in hooks.items():
        print(name)
        layer.name = name
        layer.task_id = model.model.task_id_container
        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_hook(build_bwd_hook(config.n_tasks))

    print(f'hooked {len(hooks)} modules')

