import torch
import copy
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
from mttl.models.adapters import SkilledLoRA
from mttl.models.modifiers import modify_with_routing, register_modifier
from mttl.models.modifiers.routing import (
    RouterWrapper,
    RoutingMixin,
    RoutingSelector,
    register_selector,
)


@register_selector("softmoe")
class SoftMoERouter(RoutingSelector):
    def __init__(self, config, in_d=4096):
        """
        Basic version of attention based router.
        """
        super().__init__()

        self.p = 2
        self.in_d = in_d
        self.config = config
        self.phi = nn.Parameter(
            torch.ones(self.in_d, self.config.n_skills, dtype=torch.float32)
        )

    def forward(self, routing_infos, input=None):
        bs, seq, in_d = input.shape
        x = input  # b x s x d
        # phi = self.phi[:x.shape[1],:] # s x p
        D = torch.einsum("bsd,dn->bsn", x, self.phi)  # b x s x p
        # D = torch.softmax(D, dim=1) #, torch.zeros(1, device=x.device)
        return D  # .reshape(bs, seq, self.config.n_skills, self.p) #, torch.zeros(1, device=x.device)


class RoutingLoRASoftMoe(nn.Module, RoutingMixin):
    def __init__(self, config, task_id_ptr, layer, selector=None, **kwargs):
        super().__init__()
        RoutingMixin.__init__(self, task_id_ptr)
        self.config = config
        self.in_features = layer.in_features
        self.selector = SoftMoERouter(config, in_d=self.in_features)
        # store losses and metrics
        self.losses = []
        self.metrics = {}
        self.adapter = SoftMOEAdapter(config, layer)

    def forward(self, input):
        task_id = self.routing_infos.task_ids
        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        if self.selector is not None:
            mixing_weights = self.selector(self.routing_infos, input=input)
            if isinstance(mixing_weights, tuple):
                mixing_weights, kl = mixing_weights
                self.losses.append(kl)
        else:
            bs = input.size(0)
            mixing_weights = torch.ones(
                bs, self.n_splits, self.n_skills, device=input.device, dtype=input.dtype
            )

        # self.metrics["routing"] = mixing_weights.detach().cpu().float()
        return self.adapter(input, mixing_weights, self.routing_infos)


def create_causal_prefix_mask(inst_padding_mask, padding_mask, device, bs, seq):
    # inst_padding_mask - 1s only on the instruction part, everything else is 0
    # padding_mask - 1s on the instruction and output, pad tokens are 0

    # Find the indices of the last occurrence of 1 along the last dimension
    first_one_idx = inst_padding_mask.argmax(
        dim=1, keepdim=True
    )  # start of instruction
    n_ones = inst_padding_mask.sum(dim=1).unsqueeze(-1)  # instruction length
    last_one_idx = first_one_idx + n_ones  # b x 1

    # Expand dimensions of last_ones_indices to match the shape of B
    expanded_indices = last_one_idx
    expanded_indices = expanded_indices.expand(-1, seq)  # b x s

    expanded_indices_inverse = seq - expanded_indices  # length after last 1
    expanded_indices_inverse -= torch.arange(seq).unsqueeze(0).to(device)
    expanded_indices_inverse = torch.max(
        expanded_indices_inverse, torch.zeros_like(expanded_indices_inverse)
    )
    expanded_indices_inverse = expanded_indices_inverse.flip(1)
    mask = expanded_indices + expanded_indices_inverse
    mask *= padding_mask  # 0s on the padding tokens
    mask = mask.unsqueeze(-1).expand(-1, -1, seq)
    mask = torch.einsum("bsk,bk->bsk", (mask, padding_mask))  # 0s on the padding tokens
    # shape like mask
    ar = torch.arange(seq).to(device)
    ar = ar.unsqueeze(0).unsqueeze(0).repeat(bs, seq, 1)

    A = torch.zeros(bs, seq, seq).to(mask.device)
    B = torch.ones(bs, seq, seq).to(mask.device)
    causal_padding_mask = torch.where(ar < mask, A, B)
    causal_padding_mask = 1 - causal_padding_mask  # per token mask, bs x seq x seq
    del mask, ar, A, B, expanded_indices, expanded_indices_inverse, last_one_idx
    return causal_padding_mask


class SoftMOEAdapter(SkilledLoRA):
    def __init__(self, config, layer):
        super().__init__(config, layer)

    def forward(self, input, weights, routing_infos):
        if self.training:
            self.training_steps += 1
        mixing_weights = weights
        bs, seq, n_skills = mixing_weights.size()
        # mixing_logit_tks = mixing_weights.unsqueeze(2).expand(-1, -1, seq, -1) # b x s x s x n_skills
        # causal routing: aggreage over sequence length, then aggregate over experts
        if hasattr(routing_infos, "causal_mask"):
            causal_mask = routing_infos.causal_mask
        else:
            causal_mask = create_causal_prefix_mask(
                routing_infos.inst_token_mask,
                routing_infos.pad_token_mask,
                input.device,
                bs,
                seq,
            )  # bs x s x s
            setattr(
                routing_infos, "causal_mask", causal_mask
            )  # store for next layer, we dont want to compute it over again

        # smalles number -1e38
        mixing_weight_causal = torch.where(
            causal_mask.bool(), mixing_weights, -1e38
        )  # bs x s x s x n_skills
        D = torch.softmax(
            mixing_weight_causal, dim=2
        )  # for ech token create a mixing distribution over previous tokens excluding the pad tokens
        D = torch.einsum(
            "bsk,bs->bsk", (D, routing_infos.pad_token_mask)
        )  # zero out pad tokens (there are pad tokens that have mixing_weights all -1e38, after softmax they will be non-zero)
        assert torch.isnan(D).sum() == 0
        assert torch.isclose(D[0][-1].sum(), torch.tensor(1.0))
        # apply mixing along the sequence dimension
        input_mixed = torch.einsum(
            "bsd,bps->bpd", (input, D)
        )  # b x s x n_skills x D <- mixing after forward pass, for linear operation its the same as mixng before
        # apply Loras of all skills to the mixed input
        adapter_out = torch.einsum(
            "bsd,qkdr->bsqkr", (input_mixed, self.lora_a)
        )  # bs x n_splits x n_skills x rank")
        adapter_out = torch.einsum(
            "bsqkr,qkrd->bsqkd", (adapter_out, self.lora_b)
        )  # bs x seq x n_splits x n_skills x D
        assert self.n_splits == 1
        adapter_out = adapter_out.squeeze(
            2
        )  # bs x seq x n_skills x D (D = output feaatures D)
        # ^^ expert outputs
        # transform back into b x s x D: mix along expert dimension
        del input_mixed, D, mixing_weight_causal, causal_mask
        C = torch.softmax(mixing_weights, dim=-1)  # b x s x (n_slots x n_skills)
        adapter_out = torch.einsum("bsk,bskd->bsd", (C, adapter_out))  # b x s x D
        adapter_out *= self.scaling  # / self.rank

        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup
        return self.layer(input) + adapter_out


@register_modifier("softmoe")
def modify_with_softmoe(transformer, config):
    config.router_selector = config.router_selector or "softmoe"
    config.adapter_type = config.adapter_type or "lora"

    if config.adapter_type in ["lora"]:
        return modify_with_routing(
            transformer, config, RoutingLoRASoftMoe, RouterWrapper
        )
    else:
        raise NotImplementedError(
            f"Adapter type {config.adapter_type} not implemented for softmoe modifier."
        )
