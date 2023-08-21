import torch
import copy
import torch.nn as nn   
from enum import Enum
from dataclasses import dataclass
import torch.nn.functional as F
import re 
import numpy as np       
from types import MethodType
from torch.autograd import Function
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from .utils import RoutingInfo
from projects.instr_routing import global_vars
from projects.instr_routing.models.attention import SelectAttention    

 
EPS = 1e-12


class PolytroponAdapter(nn.Module):
    @property
    def routing_infos(self) -> RoutingInfo:
        return self.task_id_ptr["routing_infos"]

              
def get_selector(config, in_d=4096):                
    from projects.instr_routing.cluster_tuning.cluster_selector import ClusterSelector

    if config.poly_selector == "poly":
        return PolytroponSelector(config)
    elif config.poly_selector == "private":
        # back-compatibility
        if config.example_to_ids_path:
            return ClusterSelector(config, soft=False)
        else:
            return PrivateSelector(config)
    elif config.poly_selector == "cluster_soft":
        return ClusterSelector(config, soft=True)
    elif config.poly_selector == "none":
        return None
        # return ClusterSelector(config, soft=True)
    elif config.poly_selector == "cluster_hard":
        return ClusterSelector(config, soft=False)
    elif config.poly_selector == "moe":
        return MoESelector(config)
    elif config.poly_selector == "x_router":
        return XRouter(config, in_d=in_d)    
    elif config.poly_selector == "attn_router":
        return AttnRouter(config, in_d=in_d)
    else:
        raise NotImplementedError()


class Selector(nn.Module):
    pass

           
class XRouter(Selector):  
    class ROUTING_OPTION(Enum):
        TOKEN = 0  # route each token seperately, i.e. each token's representation is seperately passe to the router
        INST_ONLY = 1 # train and test time we only look at instruction part
        TOKEN_SOFAR = 2 # train and test time we only look at tokens sofar       
        ALL = 3 # this is per token, but we look at all tokens sofar to calculate token's router input
        ALL_DISTILL_INST = 4 # mimic posterior teacher with prior student
    
    def __init__(self, config, in_d=4096):
        super().__init__() 
        self.config = config
        self.in_d = in_d                   
               
        self.xrouter_init_scale = config.xrouter_init_scale
        self.xrouter_sim_metric = config.xrouter_sim_metric
        self.xrouter_use_attn = config.xrouter_use_attn
        self.xrouter_x_cond = config.xrouter_x_cond    
        self.xrouter_normal_innit = config.xrouter_normal_innit
        self.xrouting_option = config.xrouting_option
        self.xrouting_sep_teacher_student = config.xrouting_sep_teacher_student
        self.xrouter_load_balancing = config.xrouter_load_balancing
        
        
        self.n_splits = config.n_splits
        self.input_layer_norm = nn.LayerNorm(in_d, dtype=global_vars.PRECISION) 
        self.ff = nn.Linear(in_d, config.n_skills * self.n_splits)   
        self.ff_router_layer_norm = nn.LayerNorm(in_d, dtype=global_vars.PRECISION)          
        self.ff_router_layer_norm.weight = nn.Parameter(torch.ones(in_d, dtype=global_vars.PRECISION)*self.xrouter_init_scale)
        
        if self.xrouter_use_attn:        
            from projects.instr_routing.models.attention import SelectAttention    
            self.xattn = SelectAttention(self.in_d, self.in_d, share_key=True, share_query=True)
            self.layer_key = nn.Parameter(torch.ones(1,1,self.in_d, dtype=global_vars.PRECISION))
        
        if not self.xrouter_x_cond:    
            self.dummy_input = nn.Parameter(torch.ones(in_d, dtype=global_vars.PRECISION))
        #innit weights and biases with normal distribution
        if self.xrouter_normal_innit:    
            self.ff.weight.data.normal_(mean=0.0, std=self.xrouter_init_scale)
            self.ff.bias.data.fill_(0)
            
        if self.xrouting_option ==4 and self.xrouting_sep_teacher_student:
            self.ff_student = nn.Linear(in_d, config.n_skills * self.n_splits, dtype=global_vars.PRECISION)   
            self.ff_student.weight.data.normal_(mean=0.0, std=self.xrouter_init_scale)
            self.ff_student.bias.data.fill_(0)
            self.ff_student_layer_norm = nn.LayerNorm(in_d, dtype=global_vars.PRECISION)              
            self.ff_student_layer_norm.weight = nn.Parameter(torch.ones(self.in_dim, dtype=global_vars.PRECISION)*self.xrouter_init_scale)
    
    @property
    def W_norm(self):   
        W = self.ff.weight
        norm = torch.norm(W, p=1, keepdim=True)
        return norm.item() #/ torch.prod(torch.tensor(W.shape))
    
    def route(self, router: nn.Linear, layer_norm: nn.LayerNorm, x):
        if self.config.xrouter_normalize_input:
            x = self.input_layer_norm(x)
        
        if self.config.xrouter_normalize_weights:  
            weights = layer_norm(router.weight)  
            return F.linear(x, weights, router.bias)
        return F.linear(x, router.weight, router.bias)
    
    def apply_mask_and_average(self, x, padding_mask):
        x_rout = x * padding_mask.unsqueeze(-1).to(x.device)
        non_zero_counts = (x_rout != 0).sum(dim=1)  
        x_rout = (x_rout.sum(dim=1) / non_zero_counts).unsqueeze(1) # same routing for each sample        
        return x_rout
               
    def forward(self,routing_infos):
        # self.xrouting_option = 4         
        bs, seq, in_d = routing_infos.x.shape
        x = routing_infos.x     
        if not self.config.xrouter_x_cond:    
            x = self.dummy_input.unsqueeze(0).repeat(bs,1).unsqueeze(1).repeat(1,seq,1) # dummy input
            
        gen_mode = 0
        x_rout = None                           
        padding_mask = routing_infos.pad_token_mask   
        if hasattr(routing_infos, "gen_mode"):
            gen_mode = routing_infos.gen_mode
            
        if self.xrouting_option>XRouter.ROUTING_OPTION.TOKEN.value:
            if gen_mode:    
                ####### GENERATION MODE #######              
                if self.xrouting_option==XRouter.ROUTING_OPTION.INST_ONLY.value:
                    if x.shape[1]==1:
                        x = self.prev_x # we do not add the generated token and always use instruction only 
                        
                if self.xrouting_option in [XRouter.ROUTING_OPTION.TOKEN_SOFAR.value, XRouter.ROUTING_OPTION.ALL.value]: 
                    if x.shape[1]==1: #we need to add cahsed previous tokens to the instructions
                        padding_mask = torch.cat((padding_mask, torch.ones(x.shape[0], (self.prev_x.shape[1]-padding_mask.shape[1])+1, device=x.device)), dim=1)
                        x = torch.cat((self.prev_x, x), dim=1)
                        
                if self.xrouting_option == XRouter.ROUTING_OPTION.ALL_DISTILL_INST.value:
                    if x.shape[1]==1:
                        x = self.prev_x # we do not add the generated token and always use instruction only 
                
                # chash previous tokens                 
                if self.xrouting_option in [XRouter.ROUTING_OPTION.INST_ONLY.value,    
                                            XRouter.ROUTING_OPTION.ALL_DISTILL_INST.value]:
                    if not x.shape[1]==1:
                        self.prev_x = copy.deepcopy(x.detach()) # use instruction for routing      
                if self.xrouting_option in [XRouter.ROUTING_OPTION.TOKEN_SOFAR.value, 
                                            XRouter.ROUTING_OPTION.ALL.value]: 
                    self.prev_x = copy.deepcopy(x.detach())
            ##########################################
            else:
                self.prev_x = None
                
            # if routings are given, use them
            if routing_infos.routings:    
                routings = routing_infos.routings.pop(0)
                return routings, torch.zeros(1, device=x.device)
            
            if x_rout is None:                          
                padding_mask = routing_infos.pad_token_mask # 1 if the token is not a pad token, so its either instruciton or output
                inst_token_mask = routing_infos.inst_token_mask if routing_infos.inst_token_mask is not None else torch.ones_like(padding_mask) # 1 if the token is part of instruction or pad token (so outputs are 0s)
                if routing_infos.inst_token_mask is None:
                    assert gen_mode 
                          
                if self.xrouting_option == XRouter.ROUTING_OPTION.ALL_DISTILL_INST.value:
                    # These were mixed up in the old codebase :O 
                    # (padding_mask * instruction_mask, padding_mask)                   
                    posterior_padding_mask = padding_mask # looks at instruction and output
                    prior_padding_mask = padding_mask * inst_token_mask # looks only on instruction
                    if hasattr(self.config, "xr4_option"):
                        if self.config.xr4_option == 'switch':
                            posterior_padding_mask = padding_mask * inst_token_mask
                            prior_padding_mask = padding_mask
                        elif self.config.xr4_option == 'teacher_output':
                            posterior_padding_mask = torch.abs(((padding_mask * inst_token_mask)-1))
                            posterior_padding_mask = posterior_padding_mask * padding_mask # output only
                            prior_padding_mask = padding_mask * inst_token_mask # instruction only
                    
                    
                    # average over tokesn that are not masked oute 
                    x_rout_prior = self.apply_mask_and_average(x, prior_padding_mask)
                    x_rout_posterior = self.apply_mask_and_average(x, posterior_padding_mask)

                    del prior_padding_mask, posterior_padding_mask
                      
                    adapter_logits_prior = self.route(self.ff, self.ff_router_layer_norm, x_rout_prior) if not self.xrouting_sep_teacher_student else self.route(self.ff_student, self.ff_student_layer_norm, x_rout_prior)
                    adapter_dist_prior = self.softmax(adapter_logits_prior/self.config.poly_selector_cluster_temp)
                    if gen_mode:
                        return adapter_dist_prior, 0.0  
                    adapter_logits_posterior = self.route(self.ff, self.ff_router_layer_norm, x_rout_posterior)
                    adapter_dist = self.softmax(adapter_logits_posterior/self.config.poly_selector_cluster_temp)
                    aux_loss = torch.zeros(1, device=x.device)
                    if self.xrouter_sim_metric == "kl":# and not gen_mode:  
                        # adapter_dist -- posterior, looks at inpiut + output. This should be P.
                        # adapter_dist_prior -- q, looks only on instruction.
                        if self.config.xrouter_reverse_kl:                   
                            # kl_divergence(p, q) -- surprise of using Q as model when true dist is P.                    
                            aux_loss = torch.distributions.kl.kl_divergence(torch.distributions.Categorical(probs=adapter_dist), 
                                                                            torch.distributions.Categorical(probs=adapter_dist_prior))  
                        else:
                            aux_loss = torch.distributions.kl.kl_divergence(torch.distributions.Categorical(probs=adapter_dist_prior), 
                                                                        torch.distributions.Categorical(probs=adapter_dist))        
                        aux_loss = aux_loss.mean()                             
                    elif self.xrouter_sim_metric == "cosine":# and not gen_mode:                        
                        aux_loss = 1-self.cosine_sim(adapter_logits_prior, adapter_logits_posterior.detach(), dim=-1)
                        aux_loss = aux_loss.mean()
                    if gen_mode or not self.training: 
                        adapter_dist = adapter_dist_prior
                    
                    if routing_infos.save_oracle_routings: 
                        routing_infos.oracle_routings.append(adapter_logits_posterior.detach())
                         
                    return adapter_dist, aux_loss

                                   
                if self.xrouting_option == XRouter.ROUTING_OPTION.INST_ONLY.value: # train and test time we only look at instruction part
                    padding_mask = padding_mask * inst_token_mask
                elif self.xrouting_option == XRouter.ROUTING_OPTION.TOKEN_SOFAR.value: # train and test time we only look at tokens sofar
                    if hasattr(routing_infos, "token_sofar_mask"): # take from cache
                        if routing_infos.token_sofar_mask is not None:
                            padding_mask = routing_infos.token_sofar_mask
                    else:        
                        padding_mask = padding_mask * inst_token_mask # only the instruction
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
                        # cache mask to prevent recalculation at each layer   
                        setattr(routing_infos, "token_sofar_mask", padding_mask)
                    assert padding_mask.dim()==3  
                    assert padding_mask.shape[0]==bs
                    assert padding_mask.shape[1]==seq
                    assert padding_mask.shape[2]==seq
                    
                      
                elif self.xrouting_option == XRouter.ROUTING_OPTION.ALL.value: # we look at all tokens, input and output (at training)
                    padding_mask = padding_mask  
                    assert padding_mask.shape[0]==bs
                    assert padding_mask.shape[1]==seq
                    
                if padding_mask.dim() == 2:          
                    # average over tokesn that are not masked out             
                    x_rout = self.apply_mask_and_average(x, padding_mask)
                elif padding_mask.dim() == 3: # different routing per token
                    x_rout = (x.unsqueeze(1) * padding_mask.unsqueeze(-1)).sum(dim=2)
                    non_zero_counts = (padding_mask != 0).sum(dim=2)
                    x_rout = (x_rout / non_zero_counts.unsqueeze(-1))
                    del non_zero_counts, padding_mask
                
                else:         
                    raise NotImplementedError()
        else:  
            x_rout = x # simple per token routing
        adapter_logits = self.route(self.ff, self.ff_router_layer_norm, x_rout)          
        adapter_probs = self.softmax(adapter_logits/self.config.poly_selector_cluster_temp)
        return adapter_probs, torch.zeros(1, device=x.device)
    
    def cosine_sim(self, x, y, dim=-1):
        if self.n_splits>1:      
            x = x.reshape(*x.shape[:-1], self.n_splits, self.config.n_skills)
            y = y.reshape(*y.shape[:-1], self.n_splits, self.config.n_skills)
            return F.cosine_similarity(x, y, dim=dim)        
        return F.cosine_similarity(x, y, dim=dim)
        
    
    def softmax(self, logit):
        if self.n_splits>1:     
            logit = logit.reshape(*logit.shape[:-1], self.n_splits, self.config.n_skills)
            out = F.softmax(logit, dim=-1)
            out = out.reshape(*logit.shape[:-2], self.config.n_skills * self.n_splits)
            return out
        return F.softmax(logit, dim=-1)
    
    
class AttnRouter(Selector): 
    def __init__(self, config, in_d=4096):
        '''
        Basic version of attention based router.
        '''
        super().__init__() 
        self.kq_dim = 32         
        self.in_d = in_d             
        self.exp_query = nn.Parameter(torch.ones(config.n_skills, self.kq_dim, dtype=global_vars.PRECISION))
        # innit from nromal
        self.exp_query.data.normal_(mean=0.0, std=0.2)
        self.xattn = SelectAttention(self.kq_dim, self.in_d, share_key=True, share_query=True)
        
    
    def forward(self, routing_infos):       
        bs, seq, in_d = routing_infos.x.shape   
        # TODO: apply mask, only attend to nstruction, ignore padding?
        padding_mask = routing_infos.pad_token_mask   
        x = routing_infos.x     
        # expert_query = self.exp_query.unsqueeze(0).repeat(bs,1,1)
        scores = self.xattn(self.exp_query, x).transpose(1,2) # bs x seq x n_skills
        assert all(scores[0].sum(dim=-1))
        # norm over experts
        i_e = scores.sum(1)    
        i_e = i_e / i_e.sum(dim=-1, keepdim=True)      
        return i_e.unsqueeze(1), torch.zeros(1, device=x.device)
            
class MoESelector(Selector):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_splits = config.n_splits
        self.n_skills = config.n_skills
        self.topk = 3

        self.module_logits = nn.Parameter(
            torch.empty((config.n_tasks, config.n_splits * config.n_skills)).uniform_(
                -1e-3, 1e-3
            )
        )

    def resize_module_logits(self, n_tasks):
        self.module_logits.data = torch.empty(
            (n_tasks, self.n_splits * self.n_skills)
        ).uniform_(-1e-3, 1e-3)

    def forward(self, routing_infos):
        module_logits = self.module_logits[routing_infos.task_ids]
        module_logits = module_logits.view(-1, self.n_splits, self.n_skills)

        if self.training:
            noise = torch.randn_like(module_logits) / self.n_skills
            module_logits = module_logits + noise

        probs = F.softmax(module_logits, dim=-1)

        top_probs, top_indices = probs.topk(self.topk, dim=-1)  # 2 active skills per task
        top_k_probs = top_probs[:, :self.topk]
        top_k_indices = top_indices[:, :self.topk]
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)

        zeros = torch.zeros_like(probs, requires_grad=True)
        module_weights = zeros.scatter(2, top_k_indices, top_k_probs)
        return module_weights


class PolytroponSelector(Selector):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_splits = config.n_splits
        self.n_skills = config.n_skills 
        self.dropout = config.module_logits_dropout   
        self.use_l2_norm = config.module_logits_l2_norm
        self.use_relaxed_bernoulli = config.module_logits_relaxed_bernoulli
        self.use_straight_through = config.module_logits_straight_through
        self.poly_average_correction = config.poly_average_correction
        self.poly_use_shared_skill = config.poly_use_shared_skill

        if self.use_relaxed_bernoulli and self.use_straight_through:
            raise ValueError("Cannot use both relaxed and straight through.")

        self.module_logits = nn.Parameter(
            torch.empty((config.n_tasks, config.n_splits * config.n_skills)).uniform_(
                -1e-3, 1e-3
            )
        )

    def resize_module_logits(self, n_tasks):
        self.module_logits.data = torch.empty(
            (n_tasks, self.n_splits * self.n_skills)
        ).uniform_(-1e-3, 1e-3)

    def forward(self, routing_infos):  
        module_logits = self.module_logits[routing_infos.task_ids]        
        module_logits = module_logits.view(-1, self.n_splits, self.n_skills)

        if self.use_l2_norm:
            module_weights = F.normalize(module_logits, p=2, dim=-1)
        else:
            if self.training and self.use_relaxed_bernoulli:
                module_logits = RelaxedBernoulli(
                    temperature=1.0, logits=module_logits
                ).rsample()
            elif self.use_straight_through:
                module_logits = torch.sigmoid(module_logits)
                module_logits_disc = torch.round(module_logits)
                # straight through estimator
                module_logits = module_logits + (module_logits_disc - module_logits).detach()
            else:
                module_logits = torch.sigmoid(module_logits)
            
            if self.dropout > 0.0:
                module_logits = nn.Dropout(self.dropout)(module_logits)

            if self.poly_use_shared_skill:
                # last skill is always active whatever the task that has been selected
                module_logits = torch.cat((
                    module_logits[:, :, :-1], module_logits[:, :, -1:] * 0.0 + 1.0
                ), dim=-1)

            if self.poly_average_correction:
                module_weights = module_logits * (np.sqrt(self.n_splits) / np.sqrt(self.n_skills))
            else:
                module_weights = module_logits / (
                    module_logits.sum(dim=-1, keepdim=True) + EPS
                )
        return module_weights


class AverageSelector(Selector):
    def __init__(self, n_skills, n_splits):
        super().__init__()

        self.n_splits = n_splits
        self.n_skills = n_skills
        self.register_buffer(
            "module_logits", torch.empty(n_splits, n_skills).fill_(1.0 / n_skills)
        )

    def forward(self, routing_infos):
        bs = routing_infos.task_ids.size(0)
        module_logits = self.module_logits.view(1, self.n_splits, self.n_skills)
        return module_logits.expand(bs, -1, -1)


class PrivateSelector(Selector):
    def __init__(self, config):
        super().__init__()

        self.n_skills = config.n_skills

    def forward(self, routing_infos):
        return F.one_hot(routing_infos.task_ids, num_classes=self.n_skills).unsqueeze(1)


class LoraAveraging(Function):
    @staticmethod             
    def forward(ctx, inputs, module_weights):
        output = torch.einsum("bqs,qsdr->bqdr", (module_weights, inputs))
        ctx.save_for_backward(module_weights)
        return output
    @staticmethod       
    def backward(ctx, grad_output):
        module_weights, = ctx.saved_tensors

        # Compute the gradients with respect to the inputs and module_weights   
        grad_inputs = torch.einsum("bqdr,qsdr->bqs", (grad_output, module_weights))
        # grad_module_weights = torch.einsum("bqs,bqdr->qsdr", (grad_output, inputs))

        return grad_inputs#, None#, grad_module_weights
        
                         
class EfficientBackwardbmm(Function):
    @staticmethod
    def forward(ctx, input, module_weights, lora_a, lora_b, in_features, rank, out_features):        
        bs = module_weights.size(0)
        ctx.rank = rank 
        ctx.lora_a = lora_a
        ctx.lora_b = lora_b        
        ctx.in_features = in_features
        ctx.out_features = out_features
        # ctx.module_weights = module_weights
        A = torch.einsum("bqs,qsdr->bqdr", (module_weights, lora_a))
        B = torch.einsum("bqs,qsrd->bqrd", (module_weights, lora_b))
        A = A.reshape(bs, in_features, rank)
        B = B.transpose(1, 2).reshape(bs, rank, out_features)
        ctx.save_for_backward(input, module_weights)#, A, B)
        return torch.bmm(input, A).bmm(B)
    
    @staticmethod       
    def backward(ctx, grad_output):            
        input,module_weights = ctx.saved_tensors   
        # retrieve saved pointers
        lora_a, lora_b = ctx.lora_a, ctx.lora_b        
        in_features, rank, out_features = ctx.in_features, ctx.rank, ctx.out_features
        module_weights = module_weights.to(dtype=lora_a.dtype)
        bs = module_weights.size(0)
        # recalculate A and B (instead of storing them)  
        A = torch.einsum("bqs,qsdr->bqdr", (module_weights, lora_a))
        B = torch.einsum("bqs,qsrd->bqrd", (module_weights, lora_b))
        A = A.reshape(bs, in_features, rank) 
        B = B.transpose(1, 2).reshape(bs, rank, out_features)
        # compute grads        
        A = A.to(dtype=grad_output.dtype)
        B = B.to(dtype=grad_output.dtype) 
        # Compute gradients with respect to the input, module_weights, lora_a, lora_b
        # grad_input is b x s x d
        grad_input = grad_output.bmm(B.transpose(1, 2)).bmm(A.transpose(1, 2))
        # grad w.r.t B, lora_b is q x s x 4 x d
        grad_B = grad_output.transpose(1, 2).bmm(torch.bmm(input, A)).transpose(1, 2)
        grad_lora_b = torch.einsum("bqs,qrd->qsrd", (module_weights, grad_B))
        # grad w.r.t A, lora_a is q x s x d x r
        grad_A = grad_output.bmm(B.transpose(1,2)).transpose(1,2).bmm(input)    
        grad_lora_a = torch.einsum("bqs,qdr->qsdr", (module_weights, grad_A)).transpose(2,3)

        return (   
            grad_input,
            None, # TODO: compute grads w.r.t. module_weights if needed.
            grad_lora_a,
            grad_lora_b,
            None,
            None,
            None,
        )   


class PolyLoRALinear(PolytroponAdapter):                          
    def __init__(self, config, task_id_ptr, linear_layer, selector=None, **kwargs):
        super().__init__()         
        self.layer_name = kwargs.get("layer_name", None)
        self.share_a = config.share_lora_a
        self.n_splits = config.n_splits
        self.n_tasks = config.n_tasks
        self.n_skills = config.n_skills                    
        self.aux_mi_loss_factor = config.aux_mi_loss_factor
        self.same_lora_init = config.same_lora_init
        self.share_lora_at_attn = config.share_lora_at_attn
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.use_warmup = config.lora_warmup
        self.rank = config.lora_rank
        self.weight = linear_layer.weight
        self.linear_layer = linear_layer
        self.bias = linear_layer.bias
        self.kaiming_init = config.lora_kaiming_init
        self.lora_randb_init = config.lora_randb_init          
               
        self.xrouter_load_balancing=config.xrouter_load_balancing
        
        self.task_id_ptr = task_id_ptr
        self.training_steps = 0.0
        self.lora_alpha = config.lora_alpha if hasattr(config, "lora_alpha") else 1.0
        self.scaling = self.lora_alpha / self.rank
        self.merge_A_B_seperately = config.merge_A_B_seperately if hasattr(config, "merge_A_B_seperately") else True
        if selector is None:
            self.selector = get_selector(config, in_d = linear_layer.in_features)
        else:
            self.selector = selector

        self.lora_a = nn.Parameter(
            self.weight.new_empty(
                self.n_splits,                
                self.n_skills if not self.share_a else 1,
                linear_layer.in_features // self.n_splits,
                self.rank,
                dtype=global_vars.PRECISION,
            )
        )       
        self.lora_b = nn.Parameter(
            self.weight.new_empty(
                self.n_splits,
                self.n_skills,
                self.rank,             
                linear_layer.out_features // self.n_splits,
                dtype=global_vars.PRECISION,
            )
        )
        self.reset_parameters() 

    def reset_parameters(self):
        import math
        n_skills_a = self.n_skills if not self.share_a else 1
        if self.kaiming_init:
            for skill in range(n_skills_a): 
                for split in range(self.n_splits):
                    param = torch.empty((self.rank, self.in_features // self.n_splits))
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    self.lora_a.data[split, skill, :, :] = param.T
        else:
            gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
            std = gain / math.sqrt(self.in_features)

            with torch.no_grad(): 
                self.lora_a.uniform_(-std, std)
                if self.same_lora_init:
                    print(self.same_lora_init)
                    # set all skills to have equal innit
                    param = self.lora_a.data[:, 0, :, :]
                    for skill in range(n_skills_a):
                        self.lora_a.data[:, skill, :, :] = param
                    if n_skills_a>1:
                        assert torch.allclose(self.lora_a.data[:, 0, :, :], self.lora_a.data[:, 1, :, :])

        # ensure that initially, adding the adapter does not change the output
        if self.use_warmup or self.lora_randb_init:
            with torch.no_grad():
                self.lora_b.uniform_(-std, std)
        else:
            torch.nn.init.zeros_(self.lora_b)

    def forward(self, input):
        if self.training:       
            self.training_steps += 1

        task_id = self.routing_infos.task_ids

        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        if self.selector is not None:          
            setattr(self.routing_infos, "x", input)       
            mixing_weights = self.selector(self.routing_infos)
            delattr(self.routing_infos, "x")
            if isinstance(mixing_weights, tuple): 
                mixing_weights, kl = mixing_weights
                self.routing_infos.aux_loss.append(kl)
            mixing_weights.to(input.device)
        else:
            bs = input.size(0)  
            mixing_weights = torch.ones(bs, self.n_splits, self.n_skills, device=input.device, dtype=input.dtype)
        if isinstance(self.selector, XRouter):          
            bs, seq, n_skills_x_n_splits = mixing_weights.size()
            # seq == 1 is per-example routing, or sdqwuence length is per-tokeb routing
            if self.n_splits>1:   
                mixing_weights = mixing_weights.reshape(bs, seq, self.n_splits, self.n_skills)
            assert seq == 1, "per token routing is not implemented yet for n_splits > 1"
        else:
            bs, n_splits, n_skills = mixing_weights.size()
        
        # rnadom probabilities
        # print(mixing_weights[0]) 
        if self.merge_A_B_seperately:
            # A is    n_splits, n_skills, D // n_splits, rank
            # we want bs,       n_splits, D // n_splits, rank                    
            A = torch.einsum("bqs,qsdr->bqdr", (mixing_weights, self.lora_a))
            B = torch.einsum("bqs,qsrd->bqrd", (mixing_weights, self.lora_b))
            A = A.reshape(bs, self.in_features, self.rank)              
            B = B.transpose(1, 2).reshape(bs, self.rank, self.out_features)
            adapter_out = input.bmm(A).bmm(B) * self.scaling # / self.rank
        else:              
            if self.n_splits>1: # MHR TODO: double check this
                s_l = input.shape[1]             
                input = input.unsqueeze(2).reshape(bs, s_l, self.n_splits, self.in_features // self.n_splits)
                adapter_out = torch.einsum("bsqd,qkdr->bsqkr", (input, self.lora_a))
                adapter_out = torch.einsum("bsqkr,qkrd->bsqkd", (adapter_out, self.lora_b)) # bs x seq x n_splits x n_skills x D//n_splits
                adapter_out = adapter_out.transpose(2,3) # bs x seq x n_skills x n_splits x D//n_splits
                assert self.n_skills>1, "n_skills must be > 1 for n_splits > 1, other is not implemented yet"
                mixing_weights = mixing_weights.squeeze(1)
                adapter_out = torch.einsum("bskqd,bqk->bsqd", (adapter_out, mixing_weights)) # bs x seq x n_splits x D//n_splits
                # stack into b x s x D: stack last dimention              
                adapter_out = adapter_out.reshape(bs, s_l, self.out_features)
                input = input.reshape(bs, s_l, self.in_features)
                
            else:
                # x * A                   
                adapter_out = torch.einsum("bsd,qkdr->bsqkr", (input, self.lora_a)) # bs x n_splits x n_skills x rank")            
                # x B
                adapter_out = torch.einsum("bsqkr,qkrd->bsqkd", (adapter_out, self.lora_b)) # bs x seq x n_splits x n_skills x D
                # x weights
                if self.n_skills>1:  # mixing_weights is bs x n_splits/seq x n_skills
                        if mixing_weights.shape[1]==1:
                            # per example routing
                            adapter_out = torch.einsum("bsqkd,bqk->bsd", (adapter_out, mixing_weights)) # bs x seq x n_splits x D
                        else:
                            # a = adapter_out * mixing_weights.unsqueeze(2).unsqueeze(-1) # bs x seq x n_splits x n_skills x D
                            # a = a.sum(dim=3).squeeze() # bs x seq x n_splits x D
                            # mixing_weights is bs x seg x n_skills, seperate routing for each seq
                            adapter_out = torch.einsum("bsqkd,bsk->bsd", (adapter_out, mixing_weights)) # bs x seq x n_splits x D
                            # a == adapter_out should be all True.
                        
                else:    
                    adapter_out = adapter_out.squeeze(2).squeeze(2) # bs x seq x D
            adapter_out *= self.scaling # / self.rank
            # adapter_weight = torch.einsum("qsdr,qsrk->qsdk", (self.lora_a, self.lora_b)) # outer product
            # adapter_weight = torch.einsum("bqs,qsrd->bqrd", (mixing_weights.detach(), adapter_weight)) # bs x n_splits x D x D
            # adapter_weight = adapter_weight.reshape(bs, self.in_features, -1)           
            # adapter_out = input.bmm(adapter_weight) * self.scaling # / self.rank
            # self.lora_b is n_splits, n_skills, rank, D // n_splits
            # input is bs, sl, D // n_splits
            
        
        # A = LoraAveraging.apply(self.lora_a, mixing_weights)
        # B = LoraAveraging.apply(self.lora_b, mixing_weights)
        
        # A = A.reshape(bs, self.in_features, self.rank)
        # B = B.transpose(1, 2).reshape(bs, self.rank, self.out_features) 
        # adapter_out = EfficientBackwardbmm.apply(input, mixing_weights.detach(), self.lora_a,    
        #                                 self.lora_b, self.in_features, self.rank, self.out_features) * self.scaling # / self.rank
        ####### KEEP TRACK of routing ########
        # track entropy of the routing distribution over last dim    
        # mixing_weights_ = mixing_weights.view(-1, self.n_skills)#.detach() # ex x n_skills        
        # mixing_weights_mean = mixing_weights.transpose(0,1).mean(dim=1) # n_splits x n_skills
        
        # average_normalized_entropy = -torch.sum(mixing_weights_ * torch.log(mixing_weights_ + EPS), dim=-1)/ np.log(self.n_skills) if self.n_skills>1 else torch.ones_like(mixing_weights_[:,0]) # ex 
        # # solit in n_splits chunks     
        # average_normalized_entropy = average_normalized_entropy.reshape(bs, self.n_splits).mean(dim=0) # bs    
        # # how different are the routinf for different examples? calculate MI, entropy of average - average entropy
        # mixing_weights_mean = mixing_weights.transpose(0,1).mean(dim=1) # n_splits x n_skills        
        # entropy_of_av_normalized = -torch.sum(mixing_weights_mean * torch.log(mixing_weights_mean + EPS), dim=-1)/ np.log(self.n_skills) if self.n_skills>1 else torch.zeros_like(mixing_weights_mean[0]) # ex
        # div = (entropy_of_av_normalized - average_normalized_entropy).mean() # mean over n_splits
             
        # if self.x_router_load_balancing:
        #     # add MI maximization as aux loss
        #     self.routing_infos.aux_loss.append(self.aux_mi_loss_factor*(1-div))       
              
        # self.routing_infos.metrics["div"].append(div.item())     
        # self.routing_infos.metrics["routing_entropy"].append(average_normalized_entropy.mean().item())
        if self.layer_name is not None:
            self.routing_infos.metrics[self.layer_name+"_routing"]=mixing_weights.detach().cpu().float()
        
        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:      
            adapter_out = adapter_out * warmup 
        # print((self.linear_layer(input) + adapter_out).shape)
        return self.linear_layer(input) + adapter_out
        # return F.linear(input, self.weight, self.bias) + adapter_out


class PolyIA3Linear(PolytroponAdapter):
    def __init__(self, config, task_id_ptr, linear_layer, selector=None, **kwargs):
        super().__init__()

        self.n_splits = config.n_splits
        self.n_tasks = config.n_tasks
        self.n_skills = config.n_skills
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.task_id_ptr = task_id_ptr

        assert self.out_features % config.n_splits == 0

        data = torch.ones(
            self.n_skills, self.n_splits, self.out_features // self.n_splits
        )
        self.lora_a = nn.Parameter(data)

        if selector is None:
            self.selector = get_selector(config)
        else:
            self.selector = selector

    def forward(self, input):
        task_id = self.routing_infos.task_ids

        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        # bs, n_splits, n_skills
        mixing_weights = self.selector(self.routing_infos)

        # n_skills, n_splits, D // n_splits
        weight = self.lora_a
        A = torch.einsum("bqs,sqd->bqd", (mixing_weights, weight))
        A = A.reshape(input.size(0), 1, -1)
        return F.linear(input, self.weight, self.bias) * A

    def extra_repr(self):
        return "n_skills={}, in_features={}, out_features={}, bias={}".format(
            self.n_skills, self.in_features, self.out_features, self.bias is not None
        )


class SkilledModel:
    @staticmethod
    def register_functions(object):
        methods = [
            method
            for method in dir(SkilledModel)
            if not method.startswith("__") and not "register_functions" in method
        ]

        for method in methods:
            print("Registering method: ", method)
            setattr(object, method, MethodType(getattr(SkilledModel, method), object))
        return object
    
    @staticmethod 
    def set_selector(object, config, selector_to_replace=PolytroponSelector, new_selector=AverageSelector):
        """Switches PolytroponSelector to AverageSelector.
        """
        for name, module in object.named_modules():   
            for name, inner_mod in module.named_children():
                if isinstance(inner_mod, selector_to_replace):
                    print(
                        f"Replacing with {new_selector}: ",
                        name,
                        "n_skills:",
                        inner_mod.n_skills,
                    )
                    n_splits = inner_mod.n_splits if hasattr(inner_mod, "n_splits") else 1
                    setattr(
                        module,
                        name,
                        new_selector(config),
                    )

    @staticmethod 
    def switch_selector_to_average(object, selector_to_replace=PolytroponSelector):
        """Switches PolytroponSelector to AverageSelector.
        """
        for name, module in object.named_modules():
            for name, inner_mod in module.named_children():
                if isinstance(inner_mod, selector_to_replace):
                    print(
                        "Replacing with average: ",
                        name,
                        "n_skills:",
                        inner_mod.n_skills,
                    )
                    n_splits = inner_mod.n_splits if hasattr(inner_mod, "n_splits") else 1
                    setattr(
                        module,
                        name,
                        AverageSelector(inner_mod.n_skills, n_splits),
                    )

    @staticmethod
    def get_adapters(object):
        adapters = {}
        for n, m in object.named_modules():
            if isinstance(m, PolytroponAdapter):
                adapters[n] = m
        return adapters

    @staticmethod
    def get_selectors(object):
        selectors = {}
        added_selectors = set()

        for name, adapter in object.get_adapters().items():
            # selectors might be shared across adapters
            if adapter.selector not in added_selectors:
                added_selectors.add(adapter.selector)
                selectors[name + ".selector"] = adapter.selector
        return selectors

    @staticmethod
    def resize_module_logits(object, n_tasks):
        """Resizes the vector routing, in case of fine-tuning.
        """
        for name, selector in object.get_selectors().items():
            print("Resizing module_logits of selector", name, "with", n_tasks, "tasks.")
            selector.resize_module_logits(n_tasks)
    
    @staticmethod
    def remove_skills(object, skill_ids_to_keep):
        print("Removing skills, keeping", skill_ids_to_keep)
        for name, adapter in object.get_adapters().items():
            if isinstance(adapter, PolyLoRALinear):
                if adapter.lora_a.shape[1]>1:
                    adapter.lora_a= nn.Parameter(adapter.lora_a[:, skill_ids_to_keep, :, :])
                if adapter.lora_b.shape[1]>1:
                    adapter.lora_b= nn.Parameter(adapter.lora_b[:, skill_ids_to_keep, :, :])
                adapter.n_skills = len(skill_ids_to_keep)
                adapter.selector.n_skills = len(skill_ids_to_keep)



def modify_with_poly(transformer, config, PolyLayer):
    
    # How to "bin" different levels of selectors ?
    def _extract_identifier(string, match_on='coder'):
        """ Returns a unique identifier for the "chunk" of layers sharing the 
        same underlying selector
        # e.g. 'block' : 'encoder.block.0.layer.0.SelfAttention' -> 'encoder.block.0'
        """
        pattern_map = {
            'coarsegrained' : None, 
            'finegrained' : None,
            'layerwise' : 'layer',
            'blockwise' : 'block',
            'coderwise' : 'coder'
        }
        assert match_on in pattern_map.keys()

        if match_on == 'finegrained':
            return string
        if match_on == 'coarsegrained': 
            return ''

        match_on = pattern_map[match_on] 
        left_idx = string.find(f'{match_on}.') + len(match_on) + 1
        right_idx = string[left_idx:].find('.') 
        return string[:left_idx + right_idx]
    
    selectors = {}
    total_layers = 0    
    n_skills = copy.deepcopy(config.n_skills) 
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):    
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    identifier = _extract_identifier(f'{m_name}.{c_name}', config.poly_granularity)
                    if identifier not in selectors.keys():
                        selectors[identifier] = get_selector(config, in_d=layer.in_features)
                    selector = selectors[identifier]
                    total_layers += 1
                    config.n_skills = n_skills
                    print(f"Patching {m_name}.{c_name}...")  
                    if "attn" in f"{m_name}.{c_name}" and config.share_lora_at_attn:
                        config.n_skills=1
                    
                    # keep track of layer information
                    layer_name = f"{m_name}.{c_name}"
                    # config.layer_name = layer_name
                        
                    setattr(
                        module,
                        c_name,
                        PolyLayer(
                            config,
                            transformer.task_id_container,
                            layer,
                            selector=selector,
                            layer_name = layer_name
                        ),
                    )

    print(f'created {len(selectors)} selectors for a total of {total_layers} adapted layers')
    return SkilledModel.register_functions(transformer)


def modify_with_poly_ia3(transformer, config):
    return modify_with_poly(transformer, config, PolyIA3Linear)


def modify_with_poly_lora(transformer, config):
    return modify_with_poly(transformer, config, PolyLoRALinear)
