import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from types import MethodType
from torch.autograd import Function
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import math
from mttl.models.utils import RoutingInfo


EPS = 1e-12


class PolytroponAdapter(nn.Module):
    """
    Adapter class for the PolytroponSelector. Returns the routing information for the current task.
    """

    @property
    def routing_infos(self) -> RoutingInfo:
        """
        Returns the routing information for the current task.

        Returns:
            RoutingInfo: The routing information for the current task.
        """
        return self.task_id_ptr["routing_infos"]


def get_selector(config):
    from mttl.cluster_tuning.cluster_selector import ClusterSelector

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
        return XRouter(config)
    else:
        raise NotImplementedError()


class Selector(nn.Module):
    pass


class XRouter(Selector):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.x_routing_option = config.x_routing_option
        self.sim_metric = config.x_router_sim_metric
        self.ff = nn.Linear(4096, config.n_skills)
        # innit weights and biases with kaiming
        if config.xrouter_kaiming:
            self.ff.weight.data.normal_(mean=0.0, std=0.02)
            self.ff.bias.data.fill_(0)
        if self.x_routing_option == 4 and config.sep_teacher_student:
            self.ff_student = nn.Linear(4096, config.n_skills)
            self.ff_student.weight.data.normal_(mean=0.0, std=0.02)
            self.ff_student.bias.data.fill_(0)

    def forward(self, routing_infos):
        # self.x_routing_option = 4
        x = routing_infos.x
        gen_mode = 0
        x_rout = None
        padding_mask = routing_infos.pad_token_mask
        if hasattr(routing_infos, "gen_mode"):
            gen_mode = routing_infos.gen_mode

        if self.x_routing_option > 0:
            if gen_mode:
                if self.x_routing_option == 1:
                    if x.shape[1] == 1:
                        x = (
                            self.prev_x
                        )  # we do not add the generated token and always use instruction only
                if self.x_routing_option in [2, 3]:
                    if (
                        x.shape[1] == 1
                    ):  # we need to add cahsed previous tokens to the instructions
                        padding_mask = torch.cat(
                            (
                                padding_mask,
                                torch.ones(
                                    x.shape[0],
                                    (self.prev_x.shape[1] - padding_mask.shape[1]) + 1,
                                    device=x.device,
                                ),
                            ),
                            dim=1,
                        )
                        x = torch.cat((self.prev_x, x), dim=1)
                if self.x_routing_option == 4:
                    if x.shape[1] == 1:
                        x = self.prev_x

                # chas previous tokens
                if self.x_routing_option in [1, 4]:
                    if not x.shape[1] == 1:
                        self.prev_x = copy.deepcopy(
                            x.detach()
                        )  # use instruction for routing
                if self.x_routing_option in [2, 3]:
                    self.prev_x = copy.deepcopy(x.detach())
            else:
                self.prev_x = None

            if x_rout is None:
                if isinstance(padding_mask, torch.Tensor):
                    if padding_mask.dim() == 2:
                        x_rout = x * padding_mask.unsqueeze(-1).to(x.device)
                        non_zero_counts = (x_rout != 0).sum(dim=1)
                        x_rout = (x_rout.sum(dim=1) / non_zero_counts).unsqueeze(
                            1
                        )  # same routing for each sample
                    elif padding_mask.dim() == 3:  # different routing per token
                        # seq = x.shape[1]
                        # # x = x.unsqueeze(1).repeat(1,seq,1,1)
                        # x_rout = x.unsqueeze(1).repeat(1,seq,1,1) * padding_mask.unsqueeze(-1)#.to(x.device)
                        # x_rout = x_rout.sum(dim=2)
                        # Element-wise multiplication with padding_mask using broadcasting
                        x_rout = (x.unsqueeze(1) * padding_mask.unsqueeze(-1)).sum(
                            dim=2
                        )
                        # del padding_mask
                        # del x
                        non_zero_counts = (padding_mask != 0).sum(dim=2)
                        x_rout = x_rout / non_zero_counts.unsqueeze(-1)
                        del non_zero_counts, padding_mask
                elif isinstance(padding_mask, tuple):
                    assert self.x_routing_option == 4
                    posterior_padding_mask = padding_mask[
                        0
                    ]  # looks at instruction and output
                    prior_padding_mask = padding_mask[1]  # looks only on instruction

                    x_rout_prior = x * prior_padding_mask.unsqueeze(-1).to(x.device)
                    non_zero_counts = (x_rout_prior != 0).sum(dim=1)
                    x_rout_prior = (
                        x_rout_prior.sum(dim=1) / non_zero_counts
                    ).unsqueeze(
                        1
                    )  # same routing for each sample

                    x_rout_posterior = x * posterior_padding_mask.unsqueeze(-1).to(
                        x.device
                    )
                    non_zero_counts = (x_rout_posterior != 0).sum(dim=1)
                    x_rout_posterior = (
                        x_rout_posterior.sum(dim=1) / non_zero_counts
                    ).unsqueeze(
                        1
                    )  # same routing for each sample

                    del non_zero_counts, prior_padding_mask, posterior_padding_mask

                    adapter_logits_prior = (
                        self.ff(x_rout_prior)
                        if not self.config.sep_teacher_student
                        else self.ff_student(x_rout_prior)
                    )
                    adapter_dist_prior = F.softmax(
                        adapter_logits_prior / self.config.poly_selector_cluster_temp,
                        dim=-1,
                    )
                    if gen_mode:
                        return adapter_dist_prior, 0.0
                    adapter_logits_posterior = self.ff(x_rout_posterior)
                    adapter_dist = F.softmax(
                        adapter_logits_posterior
                        / self.config.poly_selector_cluster_temp,
                        dim=-1,
                    )
                    aux_loss = 0.0
                    if self.sim_metric == "kl":  # and not gen_mode:
                        # adapter_dist -- posterior, looks at inpiut + output. This should be P.
                        # adapter_dist_prior -- q, looks only on instruction.
                        if self.config.reverse_xrouter_kl:
                            # kl_divergence(p, q) -- surprise of using Q as model when true dist is P.
                            aux_loss = torch.distributions.kl.kl_divergence(
                                torch.distributions.Categorical(probs=adapter_dist),
                                torch.distributions.Categorical(
                                    probs=adapter_dist_prior
                                ),
                            )
                        else:
                            aux_loss = torch.distributions.kl.kl_divergence(
                                torch.distributions.Categorical(
                                    probs=adapter_dist_prior
                                ),
                                torch.distributions.Categorical(probs=adapter_dist),
                            )
                        aux_loss = aux_loss.mean()
                    elif self.sim_metric == "cosine":  # and not gen_mode:
                        aux_loss = 1 - F.cosine_similarity(
                            adapter_logits_prior,
                            adapter_logits_posterior.detach(),
                            dim=-1,
                        )
                        aux_loss = aux_loss.mean()
                    if gen_mode or not self.training:
                        adapter_dist = adapter_dist_prior

                    return adapter_dist, aux_loss

                else:
                    raise NotImplementedError()
        else:
            x_rout = x  # simple per token routing
        # del x, padding_mask
        adapter_logits = self.ff(x_rout)
        adapter_probs = F.softmax(
            adapter_logits / self.config.poly_selector_cluster_temp, dim=-1
        )
        return adapter_probs, 0.0

    # def forward(self, routing_infos):
    #     x = routing_infos.x
    #     gen_mode=0
    #     if hasattr(routing_infos, "gen_mode"):
    #         gen_mode = routing_infos.gen_mode

    #     if self.x_routing_option > 0:
    #         padding_mask = routing_infos.pad_token_mask
    #         instruction_mask = (routing_infos.labels == -100).float()

    #         if self.x_routing_option == 1:
    #             padding_mask *= instruction_mask
    #         elif self.x_routing_option == 2 or self.x_routing_option == 3:
    #             padding_mask *= instruction_mask

    #             last_ones_indices = padding_mask.sum(dim=1, keepdim=True).cpu()
    #             bs, seq, d = x.shape
    #             ar = torch.arange(seq)

    #             mask = last_ones_indices + (seq - last_ones_indices) - ar.unsqueeze(0)
    #             mask = torch.clamp(mask, 0, seq - 1)

    #             padding_mask = (ar.unsqueeze(0) < mask).unsqueeze(-1).to(padding_mask.device)
    #             x_rout = x.unsqueeze(1) * padding_mask.to(x.device)

    #             non_zero_counts = (x_rout != 0).sum(dim=2)
    #             x_rout = (x_rout.sum(dim=2) / non_zero_counts)

    #     if gen_mode and x.shape[1] == 1:
    #         if self.x_routing_option == 1:
    #             x_rout = self.prev_x
    #         else:
    #             padding_mask = torch.cat((padding_mask, torch.ones(x.shape[0], self.prev_x.shape[1] - padding_mask.shape[1] + 1, device=x.device)), dim=1)
    #             x_rout = torch.cat((self.prev_x, x_rout), dim=1)

    #     if gen_mode:
    #         self.prev_x = copy.deepcopy(x.detach())
    #     else:
    #         self.prev_x = None

    #     if x_rout is None:
    #         padding_mask = routing_infos.pad_token_mask
    #         x_rout = x.unsqueeze(1) * padding_mask.unsqueeze(-1)
    #         non_zero_counts = (x_rout != 0).sum(dim=1)
    #         x_rout = (x_rout.sum(dim=1) / non_zero_counts).unsqueeze(1)

    #     adapter_logits = self.ff(x_rout)
    #     adapter_probs = F.softmax(adapter_logits / self.config.poly_selector_cluster_temp, dim=-1)
    #     return adapter_probs


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

        top_probs, top_indices = probs.topk(
            self.topk, dim=-1
        )  # 2 active skills per task
        top_k_probs = top_probs[:, : self.topk]
        top_k_indices = top_indices[:, : self.topk]
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
                module_logits = (
                    module_logits + (module_logits_disc - module_logits).detach()
                )
            else:
                module_logits = torch.sigmoid(module_logits)

            if self.dropout > 0.0:
                module_logits = nn.Dropout(self.dropout)(module_logits)

            if self.poly_use_shared_skill:
                # last skill is always active whatever the task that has been selected
                module_logits = torch.cat(
                    (module_logits[:, :, :-1], module_logits[:, :, -1:] * 0.0 + 1.0),
                    dim=-1,
                )

            if self.poly_average_correction:
                module_weights = module_logits * (
                    np.sqrt(self.n_splits) / np.sqrt(self.n_skills)
                )
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
        (module_weights,) = ctx.saved_tensors

        # Compute the gradients with respect to the inputs and module_weights
        grad_inputs = torch.einsum("bqdr,qsdr->bqs", (grad_output, module_weights))
        # grad_module_weights = torch.einsum("bqs,bqdr->qsdr", (grad_output, inputs))

        return grad_inputs  # , None#, grad_module_weights


class EfficientBackwardbmm(Function):
    @staticmethod
    def forward(
        ctx, input, module_weights, lora_a, lora_b, in_features, rank, out_features
    ):
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
        ctx.save_for_backward(input, module_weights)  # , A, B)
        return torch.bmm(input, A).bmm(B)

    @staticmethod
    def backward(ctx, grad_output):
        input, module_weights = ctx.saved_tensors
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
        grad_A = grad_output.bmm(B.transpose(1, 2)).transpose(1, 2).bmm(input)
        grad_lora_a = torch.einsum("bqs,qdr->qsdr", (module_weights, grad_A)).transpose(
            2, 3
        )

        return (
            grad_input,
            None,  # TODO: compute grads w.r.t. module_weights if needed.
            grad_lora_a,
            grad_lora_b,
            None,
            None,
            None,
        )


class PolyLoRALinear(PolytroponAdapter):
    def __init__(self, config, task_id_ptr, linear_layer, selector=None):
        super().__init__()
        self.same_lora_init = config.same_lora_init
        self.share_a = config.share_lora_a
        self.n_splits = config.n_splits
        self.n_tasks = config.n_tasks
        self.n_skills = config.n_skills
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
        self.task_id_ptr = task_id_ptr
        self.training_steps = 0.0
        self.lora_alpha = config.lora_alpha if hasattr(config, "lora_alpha") else 1.0
        self.scaling = self.lora_alpha / self.rank
        self.merge_A_B_seperately = (
            config.merge_A_B_seperately
            if hasattr(config, "merge_A_B_seperately")
            else True
        )
        if selector is None:
            self.selector = get_selector(config)
        else:
            self.selector = selector

        self.lora_a = nn.Parameter(
            self.weight.new_empty(
                self.n_splits,
                self.n_skills if not self.share_a else 1,
                linear_layer.in_features // self.n_splits,
                self.rank,
                dtype=torch.float32,
            )
        )
        self.lora_b = nn.Parameter(
            self.weight.new_empty(
                self.n_splits,
                self.n_skills,
                self.rank,
                linear_layer.out_features // self.n_splits,
                dtype=torch.float32,
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
                    if n_skills_a > 1:
                        assert torch.allclose(
                            self.lora_a.data[:, 0, :, :], self.lora_a.data[:, 1, :, :]
                        )

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
            mixing_weights = self.selector(self.routing_infos).type_as(self.lora_a)
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
            mixing_weights = torch.ones(
                bs, self.n_splits, self.n_skills, device=input.device, dtype=input.dtype
            ).type_as(self.lora_a)
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
            adapter_out = input.bmm(A).bmm(B) * self.scaling  # / self.rank
        else:  # ignoring n_splikts here
            # self.lora_a n_splits, n_skills, D // n_splits, rank
            # x * A
            adapter_out = torch.einsum(
                "bsd,qkdr->bsqkr", (input, self.lora_a)
            )  # bs x n_splits x n_skills x rank")
            # x B
            adapter_out = torch.einsum(
                "bsqkr,qkrd->bsqkd", (adapter_out, self.lora_b)
            )  # bs x seq x n_splits x n_skills x D
            # x weights
            if self.n_skills > 1:  # mixing_weights is bs x n_splits/seq x n_skills
                if mixing_weights.shape[1] == self.n_splits:
                    adapter_out = torch.einsum(
                        "bsqkd,bqk->bsd", (adapter_out, mixing_weights)
                    )  # bs x seq x n_splits x D
                else:
                    # a = adapter_out * mixing_weights.unsqueeze(2).unsqueeze(-1) # bs x seq x n_splits x n_skills x D
                    # a = a.sum(dim=3).squeeze() # bs x seq x n_splits x D
                    # mixing_weights is bs x seg x n_skills, seperate routing for each seq
                    adapter_out = torch.einsum(
                        "bsqkd,bsk->bsd", (adapter_out, mixing_weights)
                    )  # bs x seq x n_splits x D
                    # a == adapter_out should be all True.
            else:
                adapter_out = adapter_out.squeeze(2).squeeze(2)  # bs x seq x D
            adapter_out *= self.scaling  # / self.rank
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
        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup
        # print((self.linear_layer(input) + adapter_out).shape)
        return self.linear_layer(input) + adapter_out
        # return F.linear(input, self.weight, self.bias) + adapter_out


class PolyIA3Linear(PolytroponAdapter):
    def __init__(self, config, task_id_ptr, linear_layer, selector=None):
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
    def set_selector(
        object,
        config,
        selector_to_replace=PolytroponSelector,
        new_selector=AverageSelector,
    ):
        """Switches PolytroponSelector to AverageSelector."""
        for name, module in object.named_modules():
            for name, inner_mod in module.named_children():
                if isinstance(inner_mod, selector_to_replace):
                    print(
                        f"Replacing with {new_selector}: ",
                        name,
                        "n_skills:",
                        inner_mod.n_skills,
                    )
                    n_splits = (
                        inner_mod.n_splits if hasattr(inner_mod, "n_splits") else 1
                    )
                    setattr(
                        module,
                        name,
                        new_selector(config),
                    )

    @staticmethod
    def switch_selector_to_average(object, selector_to_replace=PolytroponSelector):
        """Switches PolytroponSelector to AverageSelector."""
        for name, module in object.named_modules():
            for name, inner_mod in module.named_children():
                if isinstance(inner_mod, selector_to_replace):
                    print(
                        "Replacing with average: ",
                        name,
                        "n_skills:",
                        inner_mod.n_skills,
                    )
                    n_splits = (
                        inner_mod.n_splits if hasattr(inner_mod, "n_splits") else 1
                    )
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
        """Resizes the vector routing, in case of fine-tuning."""
        for name, selector in object.get_selectors().items():
            print("Resizing module_logits of selector", name, "with", n_tasks, "tasks.")
            selector.resize_module_logits(n_tasks)

    @staticmethod
    def remove_skills(object, skill_ids_to_keep):
        print("Removing skills, keeping", skill_ids_to_keep)
        for name, adapter in object.get_adapters().items():
            if isinstance(adapter, PolyLoRALinear):
                if adapter.lora_a.shape[1] > 1:
                    adapter.lora_a = nn.Parameter(
                        adapter.lora_a[:, skill_ids_to_keep, :, :]
                    )
                if adapter.lora_b.shape[1] > 1:
                    adapter.lora_b = nn.Parameter(
                        adapter.lora_b[:, skill_ids_to_keep, :, :]
                    )
                adapter.n_skills = len(skill_ids_to_keep)
                adapter.selector.n_skills = len(skill_ids_to_keep)


def modify_with_poly(transformer, config, PolyLayer):
    # How to "bin" different levels of selectors ?
    def _extract_identifier(string, match_on="coder"):
        """Returns a unique identifier for the "chunk" of layers sharing the
        same underlying selector
        # e.g. 'block' : 'encoder.block.0.layer.0.SelfAttention' -> 'encoder.block.0'
        """
        pattern_map = {
            "coarsegrained": None,
            "finegrained": None,
            "layerwise": "layer",
            "blockwise": "block",
            "coderwise": "coder",
        }
        assert match_on in pattern_map.keys()

        if match_on == "finegrained":
            return string
        if match_on == "coarsegrained":
            return ""

        match_on = pattern_map[match_on]
        left_idx = string.find(f"{match_on}.") + len(match_on) + 1
        right_idx = string[left_idx:].find(".")
        return string[: left_idx + right_idx]

    selectors = {}
    total_layers = 0
    n_skills = copy.deepcopy(config.n_skills)
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    identifier = _extract_identifier(
                        f"{m_name}.{c_name}", config.poly_granularity
                    )
                    if identifier not in selectors.keys():
                        selectors[identifier] = get_selector(config)
                    selector = selectors[identifier]
                    total_layers += 1
                    config.n_skills = n_skills
                    print(f"Patching {m_name}.{c_name}...")
                    if "attn" in f"{m_name}.{c_name}" and config.share_lora_at_attn:
                        config.n_skills = 1
                    setattr(
                        module,
                        c_name,
                        PolyLayer(
                            config,
                            transformer.task_id_container,
                            layer,
                            selector=selector,
                        ),
                    )

    print(
        f"created {len(selectors)} selectors for a total of {total_layers} adapted layers"
    )
    return SkilledModel.register_functions(transformer)


class PolyLoRATensor(PolytroponAdapter):
    def __init__(self, config, task_id_ptr, linear_layer, selector=None):
        super().__init__()
        self.n_tasks = config.n_tasks
        self.n_skills = config.n_skills
        self.n_splits = config.n_splits
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.use_warmup = config.lora_warmup
        self.rank = config.lora_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.kaiming_init = config.lora_kaiming_init
        self.lora_randb_init = config.lora_randb_init
        self.task_id_ptr = task_id_ptr
        self.training_steps = 0.0

        # poly tensor-lora config.
        self.order = config.order
        self.tensor_rank = config.tensor_rank
        if selector is None:
            self.selector = get_selector(config)
        else:
            self.selector = selector

        self.embedding_dim_leaf_a = math.ceil((self.in_features) ** (1 / self.order))
        self.embedding_dim_leaf_b = math.ceil((self.out_features) ** (1 / self.order))

        self.weight_leafs_a = nn.Parameter(
            self.weight.new_empty(
                self.order,
                self.tensor_rank,
                self.rank,
                self.embedding_dim_leaf_a,
            )
        )

        self.weight_leafs_b = nn.Parameter(
            self.weight.new_empty(
                self.order,
                self.tensor_rank,
                self.rank,
                self.embedding_dim_leaf_b,
            )
        )
        # What if I just use one layer normalization
        self.layerone_normalization_a = nn.LayerNorm(
            normalized_shape=[self.rank, self.embedding_dim_leaf_a**2]
        )

        self.layerone_normalization_b = nn.LayerNorm(
            normalized_shape=[self.rank, self.embedding_dim_leaf_b**2]
        )

        self.reset_parameters()

    def reset_parameters(self):
        import math

        if self.kaiming_init:
            for skill in range(self.n_skills):
                for split in range(self.n_splits):
                    param = torch.empty((self.rank, self.in_features // self.n_splits))
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    self.lora_a.data[split, skill, :, :] = param.T
        else:
            gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
            std = gain / math.sqrt(self.in_features)

            with torch.no_grad():
                self.weight_leafs_a.uniform_(-std, std)

            with torch.no_grad():
                self.weight_leafs_b.uniform_(-std, std)

        # # ensure that initially, adding the adapter does not change the output
        # if self.use_warmup or self.lora_randb_init:
        #     with torch.no_grad():
        #         self.weight_leafs_b.uniform_(-std, std)
        # else:
        #     torch.nn.init.zeros_(self.weight_leafs_b)

    def tensor_product_construct(self, weight_leafs, embedding_dim, flag="up"):
        w = weight_leafs
        if self.order == 2:
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
            # print(w[:,:,:,:].size())
            w01 = w01.view(self.tensor_rank, self.rank, -1)
            if flag == "up":
                w01 = self.layerone_normalization_a(w01)
            elif flag == "down":
                w01 = self.layerone_normalization_b(w01)
            # print(w01.size())
            return w01[:, :, :embedding_dim]
        elif self.order == 4:
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]

            w01 = w01.view(self.tensor_rank, self.rank, -1)
            if flag == "up":
                w01 = self.layerone_normalization_a(w01)
            elif flag == "down":
                w01 = self.layerone_normalization_b(w01)
            w23 = w[2, :, :, :, None] * w[3, :, :, None, :]
            w23 = w23.view(self.tensor_rank, self.rank, -1)
            if flag == "up":
                w23 = self.layerone_normalization_a(w23)
            elif flag == "down":
                w23 = self.layerone_normalization_b(w23)

            w0123 = w01[:, :, :, None] * w23[:, :, None, :]
            w0123 = w0123.view(self.tensor_rank, self.rank, -1)
            return w0123[:, :, :embedding_dim]
        elif self.order == 8:
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
            w01 = w01.view(self.tensor_rank, self.rank, -1)
            w23 = w[2, :, :, :, None] * w[3, :, :, None, :]
            w23 = w23.view(self.tensor_rank, self.rank, -1)
            w45 = w[4, :, :, :, None] * w[5, :, :, None, :]
            w45 = w45.view(self.tensor_rank, self.rank, -1)
            w67 = w[6, :, :, :, None] * w[7, :, :, None, :]
            w67 = w67.view(self.tensor_rank, self.rank, -1)
            w0123 = w01[:, :, :, None] * w23[:, :, None, :]
            w0123 = w0123.view(self.tensor_rank, self.rank, -1)
            w4567 = w45[:, :, :, None] * w67[:, :, None, :]
            w4567 = w4567.view(self.tensor_rank, self.rank, -1)
            w01234567 = w0123[:, :, :, None] * w4567[:, :, None, :]
            w01234567 = w01234567.view(self.tensor_rank, self.rank, -1)
            return w01234567[:, :, :embedding_dim]

    def forward(self, input):
        if self.training:
            self.training_steps += 1

        task_id = self.routing_infos.task_ids

        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        mixing_weights = self.selector(self.routing_infos).to(dtype=input.dtype)
        # the number of rank equals to the rank
        bs, n_splits, n_skills = mixing_weights.size()

        self.lora_a = self.tensor_product_construct(
            self.weight_leafs_a, self.in_features, flag="up"
        )  # [tensor rank, rank, D]
        self.lora_b = self.tensor_product_construct(
            self.weight_leafs_b, self.out_features, flag="down"
        )
        self.lora_a = self.lora_a.transpose(2, 1).unsqueeze(0)
        self.lora_b = self.lora_b.unsqueeze(0)

        # A is    n_splits, n_skills, D // n_splits, rank
        # we want bs,       n_splits, D // n_splits, rank
        A = torch.einsum("bqs,qsdr->bqdr", (mixing_weights, self.lora_a))
        B = torch.einsum("bqs,qsrd->bqrd", (mixing_weights, self.lora_b))
        A = A.reshape(bs, self.in_features, self.rank)
        B = B.transpose(1, 2).reshape(bs, self.rank, self.out_features)

        adapter_out = input.bmm(A).bmm(B) / self.rank
        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup

        return F.linear(input, self.weight, self.bias) + adapter_out


def modify_with_poly_ia3(transformer, config):
    return modify_with_poly(transformer, config, PolyIA3Linear)


def modify_with_poly_tlora(transformer, config):
    return modify_with_poly(transformer, config, PolyLoRATensor)


def modify_with_poly_lora(transformer, config):
    return modify_with_poly(transformer, config, PolyLoRALinear)
