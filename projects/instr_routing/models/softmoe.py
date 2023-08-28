import torch
import copy
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
from mttl.models.adapters import SkilledLoRA
from mttl.models.modifiers import modify_with_routing, register_modifier
from mttl.models.modifiers.routing import (
    RouterWrapper,
    RoutingAdapter,
    RoutingSelector,
    get_selector,
    register_selector,
)
@register_selector("softmoe")      
class SoftMoERouter(RoutingSelector):
    def __init__(self, config, in_d=4096):
        '''
        Basic version of attention based router.
        '''    
        self.p=2
        super().__init__() 
        self.in_d = in_d    
        self.config = config   
        self.phi = nn.Parameter(torch.ones(self.in_d, self.config.n_skills, dtype=torch.float32))

    def forward(self, routing_infos, input=None):       
        bs, seq, in_d = input.shape
        x = input # b x s x d
        # phi = self.phi[:x.shape[1],:] # s x p   
        D = torch.einsum("bsd,dn->bsn", x, self.phi) # b x s x p
        # D = torch.softmax(D, dim=1) #, torch.zeros(1, device=x.device)
        return D #.reshape(bs, seq, self.config.n_skills, self.p) #, torch.zeros(1, device=x.device)
    
class RoutingLoRASoftMoe(RoutingAdapter):
    def __init__(self, config, task_id_ptr, layer, selector=None, **kwargs):
        super().__init__(task_id_ptr)
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
        return self.adapter(input, mixing_weights)

class SoftMOEAdapter(SkilledLoRA):
    def __init__(self, config, layer):
        super().__init__(config, layer)
    def forward(self, input, weights):
        if self.training:   
            self.training_steps += 1
        mixing_weights = weights
        
        
        # first apply Loras of all skills to the input and then apply averaging weights along sequence dimention
        adapter_out = torch.einsum("bsd,qkdr->bsqkr", (input, self.lora_a)) # bs x n_splits x n_skills x rank")   
        adapter_out = torch.einsum("bsqkr,qkrd->bsqkd", (adapter_out, self.lora_b)) # bs x seq x n_splits x n_skills x D
        
        

        bs, seq, n_skills = mixing_weights.size()
        mixing_weights = mixing_weights     
        mixing_logit_tks = mixing_weights.unsqueeze(2).expand(-1, -1, seq, -1) # b x s x n_slots x n_skills
        
        #causal routing
        
                       
        D = torch.softmax(mixing_weights, dim=1) #, torch.zeros(1, device=x.device)
        # D (mixing weights) is (b x s x n_skills x n_slots)
        # in original soft MOE we would do D^T X -> \in (b x (n_skills x n_slots) x D), this is per slot routing, each slot is different weighted average over seq.
        assert self.n_splits==1, "n_splits>1 is not implemented yet for SoftMoERouter"
                
        # adapter_out = adapter_out.squeeze(2) # bs x seq x n_skills x D (D = output feaatures D)
        # create examples                 
        adapter_out = adapter_out.repeat(1,1,n_slots,1,1) # b x s x n_slots x n_skills x D  
        # apply mixing weights, which is (b x s x n_slots x n_skills), defines mixing for each skill and slot  
        adapter_out = torch.einsum("bspkd,bskp->bpkd", (adapter_out, D)) # b x n_slots x n_skills x D <- mixing after forward pass, for linear operation its the same as mixng before?
        
        # transform back into b x s x D  
        mixing_weights = mixing_weights.reshape(bs, seq, self.n_skills * n_slots)
        C = torch.softmax(mixing_weights, dim=-1) # b x s x (n_slots x n_skills)
        adapter_out = adapter_out.reshape(bs, self.n_skills * n_slots, -1)
        adapter_out = torch.einsum("bsk,bkd->bsd", (C, adapter_out)) # b x s x D
        
        
               
        adapter_out *= self.scaling # / self.rank
                     
        # if self.layer_name is not None:
        #     self.routing_infos.metrics[self.layer_name+"_routing"]=mixing_weights.detach().cpu().float()
        
        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:      
            adapter_out = adapter_out * warmup 
        # print((self.linear_layer(input) + adapter_out).shape)
        return self.linear_layer(input) + adapter_out


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
            f"Adapter type {config.adapter_type} not implemented for vsmear modifier."
        )
