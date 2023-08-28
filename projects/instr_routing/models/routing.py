import torch
import copy
import torch.nn as nn   
from enum import Enum
import torch.nn.functional as F
from torch.autograd import Function

from mttl import global_vars
from mttl.models.modifiers import modify_with_routing, register_modifier
from mttl.models.modifiers.routing import RouterWrapper, RoutingAdapter, RoutingSelector, register_selector

from projects.instr_routing.models.attention import SelectAttention    


@register_selector("x_router")
class XRouter(RoutingSelector):
    class ROUTING_OPTION(Enum):
        TOKEN = 0  # route each token seperately, i.e. each token's representation is seperately passe to the router
        INST_ONLY = 1 # train and test time we only look at instruction part
        TOKEN_SOFAR = 2 # train and test time we only look at tokens sofar       
        ALL = 3 # this is per token, but we look at all tokens sofar to calculate token's router input
        ALL_DISTILL_INST = 4 # mimic posterior teacher with prior student

    def __init__(self, config, in_d):
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
        self.xrouter_x4target_detach = config.xrouter_x4target_detach
        self.xrouter_x4_target = config.xrouter_x4_target

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

    def forward(self, routing_infos, input):
        bs, seq, in_d = input.shape
        x = input   

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
                    posterior_padding_mask = padding_mask # looks at instruction and output  
                    prior_padding_mask = padding_mask * inst_token_mask # looks only on instruction                    
                    
                    # average over tokesn that are not masked oute 
                    x_rout_prior = self.apply_mask_and_average(x, prior_padding_mask)
                    x_rout_posterior = self.apply_mask_and_average(x, posterior_padding_mask)

                    del prior_padding_mask, posterior_padding_mask
                      
                    adapter_logits_prior = self.route(self.ff, self.ff_router_layer_norm, x_rout_prior) if not self.xrouting_sep_teacher_student else self.route(self.ff_student, self.ff_student_layer_norm, x_rout_prior)
                    adapter_dist_prior = self.softmax(adapter_logits_prior/self.config.router_selector_cluster_temp)
                    if gen_mode:              
                        return adapter_dist_prior, 0.0  
                    adapter_logits_posterior = self.route(self.ff, self.ff_router_layer_norm, x_rout_posterior)
                    adapter_dist_post = self.softmax(adapter_logits_posterior/self.config.router_selector_cluster_temp)
                    aux_loss = torch.zeros(1, device=x.device)
                              
                    if self.xrouter_sim_metric == "kl":# and not gen_mode:  
                        if self.config.xrouter_reverse_kl:                   
                            # kl_divergence(p, q) -- surprise of using Q as model when true dist is P.                          
                            aux_loss = torch.distributions.kl.kl_divergence(torch.distributions.Categorical(probs=adapter_dist_post), 
                                                                            torch.distributions.Categorical(probs=adapter_dist_prior))  
                        else:
                            aux_loss = torch.distributions.kl.kl_divergence(torch.distributions.Categorical(probs=adapter_dist_prior), 
                                                                        torch.distributions.Categorical(probs=adapter_dist_post))        
                        aux_loss = aux_loss.mean()                             
                        
                    elif self.xrouter_sim_metric == "cosine":# and not gen_mode:       
                        if self.xrouter_x4_target=="posterior":
                            target = adapter_logits_posterior  
                            student_logit = adapter_logits_prior
                        elif self.xrouter_x4_target=="prior":
                            target = adapter_logits_prior        
                            student_logit = adapter_logits_posterior
                        else:
                            raise NotImplementedError()
                         
                        aux_loss = 1-self.cosine_sim(student_logit, target.detach() if self.xrouter_x4target_detach else target, dim=-1)
                        aux_loss = aux_loss.mean()
                    
                    
                    if routing_infos.save_oracle_routings: 
                        routing_infos.oracle_routings.append(adapter_dist_post.detach())
                        
                    if gen_mode or not self.training:    
                        # at validation time we use prior
                        return adapter_dist_prior, aux_loss
                    
                    if self.xrouter_x4_target=="posterior":
                        return adapter_dist_post, aux_loss
                    elif self.xrouter_x4_target=="prior":
                        return adapter_dist_prior, aux_loss
                    else:
                        raise NotImplementedError()
                
                ''' 
                x_prior = [X].mean(seq)
                x _post = [X,Y].mean(seq)
                logits_post = Router(x_post)
                logits_prior = Router(x_prior)
                
                # correct way (option 1): target = posterior, detach()
                
                aux_loss = 1-cos_sim(logits_prior, logits_post.detach())
                return softmax(logits_post)
                
                # option 2: target = posterior, no detach()
                
                aux_loss = 1-cos_sim(logits_prior, logits_post)
                return softmax(logits_post)
                    
                # option 3: target = prior, detach()       <- same as below?
                aux_loss = 1-cos_sim(logits_post, logits_prior.detach())
                return softmax(logits_prior)
                    
                # option 4: target = prior, no detach() <- alrady tried somewhere?
                aux_loss = 1-cos_sim(logits_post, logits_prior)
                return softmax(logits_prior)                
                '''
                
                # old version with bug
                # if self.xrouting_option == XRouter.ROUTING_OPTION.ALL_DISTILL_INST.value:
                #     # These were mixed up in the old codebase :O 
                #     # (padding_mask * instruction_mask, padding_mask)                   
                #     # posterior_padding_mask = padding_mask # looks at instruction and output
                #     # prior_padding_mask = padding_mask * inst_token_mask # looks only on instruction
                    
                #     posterior_padding_mask = padding_mask * inst_token_mask
                #     prior_padding_mask = padding_mask  
                #     # if hasattr(self.config, "xr4_option"):
                #     #     if self.config.xr4_option == 'switch':
                #     #         posterior_padding_mask = padding_mask * inst_token_mask
                #     #         prior_padding_mask = padding_mask
                #     #     elif self.config.xr4_option == 'teacher_output':
                #     #         posterior_padding_mask = torch.abs(((padding_mask * inst_token_mask)-1))
                #     #         posterior_padding_mask = posterior_padding_mask * padding_mask # output only
                #     #         prior_padding_mask = padding_mask * inst_token_mask # instruction only
                    
                    
                #     # average over tokesn that are not masked oute 
                #     x_rout_prior = self.apply_mask_and_average(x, prior_padding_mask)
                #     x_rout_posterior = self.apply_mask_and_average(x, posterior_padding_mask)

                #     del prior_padding_mask, posterior_padding_mask
                      
                #     adapter_logits_prior = self.route(self.ff, self.ff_router_layer_norm, x_rout_prior) if not self.xrouting_sep_teacher_student else self.route(self.ff_student, self.ff_student_layer_norm, x_rout_prior)
                #     adapter_dist_prior = self.softmax(adapter_logits_prior/self.config.router_selector_cluster_temp)
                #     if gen_mode:
                #         return adapter_dist_prior, 0.0  
                #     adapter_logits_posterior = self.route(self.ff, self.ff_router_layer_norm, x_rout_posterior)
                #     adapter_dist = self.softmax(adapter_logits_posterior/self.config.router_selector_cluster_temp)
                #     aux_loss = torch.zeros(1, device=x.device)
                #     if self.xrouter_sim_metric == "kl":# and not gen_mode:  
                #         # adapter_dist -- posterior, looks at inpiut + output. This should be P.
                #         # adapter_dist_prior -- q, looks only on instruction.
                #         if self.config.xrouter_reverse_kl:                   
                #             # kl_divergence(p, q) -- surprise of using Q as model when true dist is P.                    
                #             aux_loss = torch.distributions.kl.kl_divergence(torch.distributions.Categorical(probs=adapter_dist), 
                #                                                             torch.distributions.Categorical(probs=adapter_dist_prior))  
                #         else:
                #             aux_loss = torch.distributions.kl.kl_divergence(torch.distributions.Categorical(probs=adapter_dist_prior), 
                #                                                         torch.distributions.Categorical(probs=adapter_dist))        
                #         aux_loss = aux_loss.mean()                             
                #     elif self.xrouter_sim_metric == "cosine":# and not gen_mode:                        
                #         aux_loss = 1-self.cosine_sim(adapter_logits_prior, adapter_logits_posterior.detach(), dim=-1)
                #         aux_loss = aux_loss.mean()
                #     if gen_mode or not self.training: 
                #         adapter_dist = adapter_dist_prior
                    
                #     if routing_infos.save_oracle_routings: 
                #         routing_infos.oracle_routings.append(adapter_logits_posterior.detach())
                         
                #     return adapter_dist, aux_loss

                if self.xrouting_option == XRouter.ROUTING_OPTION.INST_ONLY.value: # train and test time we only look at instruction part
                    if padding_mask.shape[1] != inst_token_mask.shape[1]:
                        breakpoint()
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
        adapter_probs = self.softmax(adapter_logits/self.config.router_selector_cluster_temp)
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
    

@register_selector("attn_router")                
class AttnRouter(RoutingSelector): 
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

    def forward(self, routing_infos, input):
        bs, seq, in_d = input.shape
        x = input

        padding_mask = routing_infos.pad_token_mask
        inst_token_mask = routing_infos.inst_token_mask if routing_infos.inst_token_mask is not None else torch.ones_like(padding_mask)
        
        # basic version:     
        # the router only attends over instruction
        padding_mask = padding_mask * inst_token_mask
        x = x * padding_mask.unsqueeze(-1)
        
        # expert_query = self.exp_query.unsqueeze(0).repeat(bs,1,1)
        scores = self.xattn(self.exp_query, x).transpose(1,2) # bs x seq x n_skills
        assert all(scores[0].sum(dim=-1))
        # norm over experts
        i_e = scores.sum(1)    
        i_e = i_e / i_e.sum(dim=-1, keepdim=True)      
        return i_e.unsqueeze(1), torch.zeros(1, device=x.device)


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


class RoutingLoRALinear(RoutingAdapter):                          
    def __init__(self, config, task_id_ptr, linear_layer, selector=None, **kwargs):
        super().__init__()

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
        self.xrouter_load_balancing = config.xrouter_load_balancing

        self.task_id_ptr = task_id_ptr
        self.training_steps = 0.0
        self.lora_alpha = config.lora_alpha if hasattr(config, "lora_alpha") else 1.0
        self.scaling = self.lora_alpha / self.rank
        self.merge_A_B_seperately = config.merge_A_B_seperately if hasattr(config, "merge_A_B_seperately") else True

        if selector is None:
            self.selector = get_selector(config, in_d=linear_layer.in_features)
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

        # store losses and metrics
        self.losses = []
        self.metrics = {}
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
            mixing_weights = self.selector(self.routing_infos, input=input)

            if isinstance(mixing_weights, tuple): 
                mixing_weights, kl = mixing_weights
                self.losses.append(kl)

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
              
        # self.metrics["div"].append(div.item())     
        # self.metrics["routing_entropy"].append(average_normalized_entropy.mean().item())

        self.metrics["routing"] = mixing_weights.detach().cpu().float()

        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:      
            adapter_out = adapter_out * warmup 

        return self.linear_layer(input) + adapter_out


@register_modifier("routing_lora")
def modify_with_routing_lora(transformer, config):
    return patch_with_routing(transformer, config, RoutingLoRALinear, RouterWrapper)
