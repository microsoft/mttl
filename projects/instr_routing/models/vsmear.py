import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mttl.models.modifiers import register_modifier
from mttl.models.modifiers.lora import SkilledLoRA, SkilledLoRA_MergeLoraAfterOP
from mttl.models.modifiers.routing import (
    RouterWrapper,
    RouterModifyMixin,
    modify_with_routing,
    RoutingMixin,
    RoutingSelector,
    get_selector,
    register_selector,
)

from projects.instr_routing.models.clm import AugmentedRoutingInfo


class Metrics:
    def __init__(self) -> None:
        self.metrics = {}

    def add(self, key, value):
        self.metrics[key] = value.detach().cpu()

    def __setitem__(self, key, value):
        self.add(key, value)

    def clear(self):
        self.metrics.clear()

    def items(self):
        return self.metrics.items()


@register_selector("smear")
class SMEARRouter(RoutingSelector):
    def __init__(self, config, in_d):
        super().__init__(config)

        self.config = config
        self.in_d = in_d
        self.n_skills = config.n_skills
        self.n_splits = config.n_splits
        self.split_dim = in_d // self.n_splits
        self.temperature = config.router_temperature
        self.normalize_weights = config.router_normalize_weights
        assert in_d % self.n_splits == 0

        # during generation we want to cache the input encoding
        # because in successive generation steps, the input is going to be only the encoding for the last token
        self._router_input_cache = None

        # TODO: check if this view messes up the initialization
        self.prior_router = nn.Linear(
            self.split_dim, self.n_splits * self.n_skills, bias=True
        )
        self.prior_router_ln = nn.LayerNorm(in_d)
        self.prior_router_ln.weight = nn.Parameter(torch.ones(in_d))

        self.router_center_momentum = self.config.router_center_momentum
        self.register_buffer("center", torch.zeros(self.n_splits, self.n_skills))

        if self.config.smear_gaussian_init:
            # normal innit
            self.prior_router.weight.data.normal_(mean=0.0, std=0.02)
            self.prior_router.bias.data.fill_(0)

        self.metrics = Metrics()

    @torch.no_grad()
    def apply_center_update(self, routes):
        if self.training:
            self.center = (
                self.center * (1 - self.router_center_momentum)
                + torch.mean(routes, dim=0, keepdim=True) * self.router_center_momentum
            )

    def route_maybe_center(
        self, input, router, router_ln, temperature=1.0, center=False
    ):
        """Computes routes for the teacher / posterior."""
        input = router_ln(input)
        input = input.view(-1, self.n_splits, self.split_dim)

        # weights : b s
        weight = router.weight.view(self.split_dim, self.n_splits, self.n_skills)
        if router.bias is None:
            bias = 0.0
        else:
            bias = router.bias.view(1, self.n_splits, self.n_skills)

        if self.normalize_weights:
            weight = weight / torch.norm(weight, p=2, dim=-1, keepdim=True)
        routes = torch.einsum("bsd,dsk->bsk", input, weight) + bias
        if center:
            self.apply_center_update(routes)
            routes = routes - self.center
        # teacher centering and sharpening
        return routes / temperature

    def apply_mask_and_average(self, x, padding_mask):
        x_rout = x * padding_mask.unsqueeze(-1).to(x.device)
        lengths = padding_mask.ne(0).sum(dim=1, keepdim=True) + 1e-5
        x_rout = x_rout.sum(dim=1) / lengths
        return x_rout

    def _get_router_inputs(
        self, input: torch.Tensor, routing_infos: AugmentedRoutingInfo
    ):
        """When generating, the successive forward calls only receive the last token (bs, 1, d).

        Therefore, at the first forward call (context encoding), we need to cache the input encodings
        so that we can use it to compute the prior routing probabilities.
        """

        repeat = 1
        if getattr(routing_infos, "encoder_output", None) is not None:
            repeat = routing_infos.task_ids.size(
                0
            ) // routing_infos.encoder_output.size(0)
            input = routing_infos.encoder_output

        router_input = routing_infos.inputs_cache_for_generation.get(self)
        if router_input is None:
            router_input = self.apply_mask_and_average(
                input,
                routing_infos.inst_token_mask,  # only instruction is marked with 1
            )

            router_input = router_input.repeat_interleave(
                repeat, dim=0
            )  # no-op if repeat==1

            # if in generation mode, cache the input encoding for the next forward calls
            if routing_infos.generation_mode:
                routing_infos.inputs_cache_for_generation[self] = router_input
        return router_input

    def forward(self, routing_infos: AugmentedRoutingInfo, input: torch.Tensor):
        self.metrics.clear()

        prior_input = self._get_router_inputs(input, routing_infos)
        prior_routes = self.route_maybe_center(
            prior_input,
            self.prior_router,
            self.prior_router_ln,
            temperature=self.temperature,
            center=self.router_center_momentum > 0.0,
        )

        routing_probs = F.softmax(prior_routes, dim=-1)
        h_pri = -(routing_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
        self.routings = routing_probs.detach().cpu()
        self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
        return routing_probs


@register_selector("vsmear")
class VSMEARRouter(SMEARRouter):
    def __init__(self, config, in_d):
        super().__init__(config, in_d)

        self.router_shared_weights = config.router_shared_weights

        if self.router_shared_weights:
            self.post_router = self.prior_router
            self.post_router_ln = self.prior_router_ln
        else:
            self.post_router = nn.Linear(
                in_d, config.n_skills * self.n_splits, bias=True
            )
            self.post_router_ln = nn.LayerNorm(in_d)
            self.post_router_ln.weight = nn.Parameter(torch.ones(in_d))

            if self.config.smear_gaussian_init:
                self.post_router.weight.data.normal_(mean=0.0, std=0.02)
                self.post_router.bias.data.fill_(0)

        self.router_teacher_ent_factor = self.config.router_teacher_ent_factor
        self.router_teacher_temperature = self.config.router_teacher_temperature
        self.router_center_momentum = config.router_center_momentum

    def get_posterior_input(self, input, routing_infos):
        padding_mask = routing_infos.pad_token_mask
        post_input = self.apply_mask_and_average(input, padding_mask)
        return post_input

    def forward(self, routing_infos: AugmentedRoutingInfo, input: torch.Tensor):
        self.metrics.clear()

        prior_input = self._get_router_inputs(input, routing_infos)

        # do not center the student, center only the teacher now
        prior_routes = self.route_maybe_center(
            prior_input,
            self.prior_router,
            self.prior_router_ln,
            temperature=self.temperature,
            center=False,
        )

        if self.training:
            # during training :-)
            assert (
                not routing_infos.generation_mode
            ), "We are not expecting to be in generation mode during training."

            post_input = self.get_posterior_input(input, routing_infos)
            post_routes = self.route_maybe_center(
                post_input,
                self.post_router,
                self.post_router_ln,
                temperature=self.router_teacher_temperature,
                center=self.router_center_momentum > 0.0,
            )
            post_probs = routing_probs = F.softmax(post_routes, dim=-1)
            prior_probs = F.softmax(prior_routes, dim=-1)

            # compute auxiliary loss (KL divergence), KL = - H(posterior) + Xent(posterior, prior)
            h_post = -(post_probs * F.log_softmax(post_routes, -1)).sum(1).mean()
            h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
            x_ent = -(post_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()

            self.routings = post_probs.detach().cpu()
            self.metrics["h_post"] = h_post / math.log(self.n_skills)
            self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
            self.metrics["x_ent"] = x_ent

            if self.router_kl_func == "kl":
                self.auxiliary_loss = self.router_kl_factor * (-h_post + x_ent)
            elif self.router_kl_func == "l2":
                self.auxiliary_loss = (
                    self.router_kl_factor * (post_routes - prior_routes).pow(2.0).mean()
                )
        else:
            # during eval :-(
            prior_probs = routing_probs = F.softmax(prior_routes, dim=-1)
            h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()

            self.routings = prior_probs.detach().cpu()
            self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
            self.auxiliary_loss = h_pri.sum() * 0.0
        return routing_probs


@register_selector("task_vsmear")
class TaskVSMEARRouter(SMEARRouter):
    """Predict Task from Prior distribution"""

    def __init__(self, config, in_d):
        super(RoutingSelector, self).__init__()
        assert (
            config.n_tasks > 0
        ), "`TaskVSMEARRouter` must be used with multitask datasets."

        self.config = config
        self.prior = nn.Linear(in_d, config.n_tasks)

        self.posterior_router = nn.Parameter(
            torch.empty((config.n_tasks, config.n_splits * config.n_skills)).uniform_(
                -1e-3, 1e-3
            )
        )

        # during generation we want to cache the input encoding
        # because in successive generation steps, the input is going to be only the encoding for the last token
        self._router_input_cache = None

        self.n_splits = 1
        self.split_dim = in_d
        self.n_tasks = self.n_skills = config.n_tasks
        self.prior_ln = nn.Identity()
        # self.prior_router_ln = nn.LayerNorm(in_d)
        # self.prior_router_ln.weight = nn.Parameter(torch.ones(in_d))

        self.adapter_splits, self.adapter_skills = config.n_splits, config.n_skills

        self.temperature = config.router_temperature
        self.normalize_weights = config.router_normalize_weights
        self.router_center_momentum = config.router_center_momentum
        self.register_buffer("center", torch.zeros(1, config.n_tasks))

        if config.smear_gaussian_init:
            # normal innit
            self.prior.weight.data.normal_(mean=0.0, std=0.02)
            self.prior.bias.data.fill_(0)

        self.metrics = Metrics()

    def forward(self, routing_infos: AugmentedRoutingInfo, input: torch.Tensor):
        self.metrics.clear()

        if getattr(routing_infos, "encoder_output", None) is not None:
            input = routing_infos.encoder_output

        prior_input = self._get_router_inputs(input, routing_infos)
        bs = prior_input.size(0)

        if self.config.task_vsmear_detach_prior_input:
            prior_input = prior_input.detach()

        # do not center the student, center only the teacher now
        prior_routes = self.route_maybe_center(
            prior_input,
            self.prior,
            self.prior_ln,
            temperature=self.temperature,
            center=False,
        )

        prior_task_probs = F.softmax(prior_routes, dim=-1).view(
            bs, self.n_tasks
        )  # (bs, n_tasks)
        task_skill_dist = self.posterior_router.view(
            -1, self.adapter_splits, self.adapter_skills
        ).sigmoid()
        task_skill_dist = task_skill_dist / task_skill_dist.sum(-1, keepdim=True)

        prior_probs = torch.einsum("bt,tsk->bsk", prior_task_probs, task_skill_dist)
        prior_log_probs = prior_probs.log()

        h_pri = -(prior_probs * prior_log_probs).sum(-1).mean()
        h_task_pri = (
            -(prior_task_probs * prior_task_probs.clamp(min=1e-30).log()).sum(-1).mean()
        )
        self.metrics["h_pri"] = h_pri / math.log(self.adapter_skills)
        self.metrics["h_task_pri"] = h_task_pri / math.log(self.n_tasks)

        if self.training:
            # during training :-)
            assert (
                not routing_infos.generation_mode
            ), "We are not expecting to be in generation mode during training."

            # Do a MHR forward pass
            post_probs = task_skill_dist[routing_infos.task_ids]
            post_log_probs = post_probs.log()

            # compute auxiliary loss (KL divergence), KL = - H(posterior) + Xent(posterior, prior)
            h_post = -(post_probs * post_log_probs).sum(-1).mean()
            x_ent = -(post_probs * prior_log_probs).sum(-1).mean()

            self.routings = post_probs.detach().cpu()
            self.metrics["h_post"] = h_post / math.log(self.adapter_skills)
            self.metrics["x_ent"] = x_ent

            # TODO: use cross-entropy with task id as label
            self.auxiliary_loss = self.config.task_vsmear_aux_lambda * F.cross_entropy(
                prior_routes.flatten(0, 1), routing_infos.task_ids
            )
            routing_probs = post_probs
        else:
            routing_probs = prior_probs
            self.routings = prior_probs.detach().cpu()
            self.auxiliary_loss = h_pri.sum() * 0.0
        return routing_probs


@register_selector("smear_pt")
class SMEARRouterPerToken(SMEARRouter):
    def __init__(self, config, in_d):
        super().__init__(config, in_d)

    def _get_router_inputs(
        self, input: torch.Tensor, routing_infos: AugmentedRoutingInfo
    ):
        return input

    def forward(self, routing_infos: AugmentedRoutingInfo, input: torch.Tensor):
        self.metrics.clear()

        prior_input = self._get_router_inputs(input, routing_infos)  # b x seq x d
        prior_routes = self.route_maybe_center(
            prior_input,
            self.prior_router,
            self.prior_router_ln,
            temperature=self.temperature,
            center=self.router_center_momentum > 0.0,
        )  # b x seq x self.n_skills

        routing_probs = F.softmax(prior_routes, dim=-1)  # b x seq x d
        h_pri = -(routing_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
        self.routings = routing_probs.detach().cpu().view(-1, self.n_skills)
        self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
        return routing_probs.unsqueeze(1)


@register_modifier("smear")
class AuxRoutingLoRALinear(SkilledLoRA, RoutingMixin, RouterModifyMixin):
    def __init__(self, config, task_id_ptr, layer, selector=None, **kwargs):
        RoutingMixin.__init__(self, task_id_ptr)
        SkilledLoRA.__init__(self, config, layer, **kwargs)

        if selector is None:
            self.selector = get_selector(config, in_d=self.in_features)
        else:
            self.selector = selector

    def forward(self, input):
        iput_dt = input.dtype
        input = input.to(torch.float32)

        task_id = self.routing_infos.task_ids
        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        if self.selector is not None:
            mixing_weights = self.selector(self.routing_infos, input=input)
        else:
            bs = input.size(0)
            mixing_weights = torch.ones(
                bs, self.n_splits, self.n_skills, device=input.device, dtype=input.dtype
            )
        output = SkilledLoRA.forward(self, input.to(iput_dt), mixing_weights)
        output = output.to(iput_dt)
        return output

    @classmethod
    def modify_transformer(cls, transformer, config):
        config.router_selector = config.router_selector or "smear"

        return modify_with_routing(cls, transformer, config, RouterWrapper)


@register_modifier("smear_pt")
class AuxRoutingLoRALinear_XR1_PT(SkilledLoRA, RoutingMixin):
    @classmethod
    def modify_transformer(cls, transformer, config):
        config.router_selector = "smear_pt"

        return modify_with_routing(cls, transformer, config, RouterWrapper)


@register_modifier("vsmear")
class VSmear_AuxRoutingLoRALinear(AuxRoutingLoRALinear):
    @classmethod
    def modify_transformer(cls, transformer, config):
        config.router_selector = "vsmear"

        return modify_with_routing(cls, transformer, config, RouterWrapper)


@register_modifier("task_vsmear")
class TaskVSmear_AuxRoutingLoRALinear(AuxRoutingLoRALinear):
    @classmethod
    def modify_transformer(cls, transformer, config):
        config.router_selector = "task_vsmear"

        return modify_with_routing(cls, transformer, config, RouterWrapper)


@register_selector("vsmear_xr4")
class VSMEARRouterExperimental(VSMEARRouter):
    """
    Adds aux loss between prior and posterior routings.
    """

    def __init__(self, config, in_d):
        super().__init__(config, in_d)
        self.prior_router_ln = nn.Identity()
        self.post_router_ln = nn.Identity()
        self.xrouter_x4_target = (
            config.xrouter_x4_target
            if hasattr(config, "xrouter_x4_target")
            else "prior"
        )
        self.xrouter_x4target_detach = (
            config.xrouter_x4target_detach
            if hasattr(config, "xrouter_x4target_detach")
            else True
        )

    def forward(self, routing_infos, input: torch.Tensor):
        self.metrics.clear()
        prior_input = self._get_router_inputs(input, routing_infos)

        # do not center the student, center only the teacher now
        prior_routes = self.route_maybe_center(
            prior_input,
            self.prior_router,
            self.prior_router_ln,
            temperature=self.temperature,
            center=False,
        )
        if self.training:
            # during training :-)
            assert (
                not routing_infos.generation_mode
            ), "We are not expecting to be in generation mode during training."

            post_input = self.get_posterior_input(input, routing_infos)
            post_routes = self.route_maybe_center(
                post_input,
                self.post_router,
                self.post_router_ln,
                temperature=self.router_teacher_temperature,
                center=self.router_center_momentum > 0.0,
            )
            post_probs = F.softmax(post_routes, dim=-1)
            prior_probs = F.softmax(prior_routes, dim=-1)

            if self.xrouter_x4_target == "posterior":
                target = post_routes * self.router_teacher_temperature
                student_logit = prior_routes * self.temperature
                routing_probs = post_probs
            elif self.xrouter_x4_target == "prior":
                target = prior_routes * self.temperature
                student_logit = post_routes * self.router_teacher_temperature
                routing_probs = prior_probs

            self.auxiliary_loss = (
                1
                - F.cosine_similarity(
                    student_logit,
                    target.detach() if self.xrouter_x4target_detach else target,
                    dim=-1,
                ).mean()
            )

            h_post = -(post_probs * F.log_softmax(post_routes, -1)).sum(1).mean()
            h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
            x_ent = -(post_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
            self.routings = routing_probs.detach().cpu()
            self.metrics["h_post"] = h_post / math.log(self.n_skills)
            self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
            self.metrics["x_ent"] = x_ent

        else:
            # during eval :-(
            prior_probs = routing_probs = F.softmax(prior_routes, dim=-1)
            h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()

            self.routings = prior_probs.detach().cpu()
            self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
            self.auxiliary_loss = h_pri.sum() * 0.0
        return routing_probs.unsqueeze(1)


@register_selector("smear_oracle")
class VSMEARRouterOracle(VSMEARRouter):
    """
    Use posterior routings for trainign and test.
    """

    def __init__(self, config, in_d):
        super().__init__(config, in_d)
        self.prior_router_ln = nn.Identity()
        self.post_router_ln = nn.Identity()

    def forward(self, routing_infos, input: torch.Tensor):
        self.metrics.clear()
        prior_input = self._get_router_inputs(input, routing_infos)

        # do not center the student, center only the teacher now
        prior_routes = self.route_maybe_center(
            prior_input,
            self.prior_router,
            self.prior_router_ln,
            temperature=self.temperature,
            center=False,
        )

        post_input = self.get_posterior_input(input, routing_infos)
        post_routes = self.route_maybe_center(
            post_input,
            self.post_router,
            self.post_router_ln,
            temperature=self.router_teacher_temperature,
            center=self.router_center_momentum > 0.0,
        )
        routing_probs = post_probs = F.softmax(post_routes, dim=-1)
        prior_probs = F.softmax(prior_routes, dim=-1)  # route with posterior

        h_post = -(post_probs * F.log_softmax(post_routes, -1)).sum(1).mean()
        h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
        x_ent = -(post_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
        self.routings = routing_probs.detach().cpu()
        self.metrics["h_post"] = h_post / math.log(self.n_skills)
        self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
        self.metrics["x_ent"] = x_ent
        self.auxiliary_loss = h_pri.sum() * 0.0
        return routing_probs.unsqueeze(1)


@register_modifier("vsmear_xr4")
class AuxRoutingLoRALinear_MergeAfterOP(SkilledLoRA_MergeLoraAfterOP, RoutingMixin):
    def __init__(self, config, task_id_ptr, layer, selector=None, **kwargs):
        RoutingMixin.__init__(self, task_id_ptr)
        SkilledLoRA_MergeLoraAfterOP.__init__(self, config, layer, **kwargs)

        if selector is None:
            self.selector = get_selector(config, in_d=self.in_features)
        else:
            self.selector = selector

        self._losses = []
        self._metrics = {}

    @property
    def losses(self):
        return self._losses

    @property
    def metrics(self):
        return self._metrics

    def clear(self):
        self._losses.clear()
        self._metrics.clear()

    def forward(self, input):
        # we follow the rule: upcast the inputs and downcast the outputs.
        # when e.g. evaluating in float16, since Loras and router operate in floar32,
        # we need to upcast the inputs and then downcast the outputs.
        # In the adapter, we might need to downcast the adapter input back to the adapter's type (e.g. float16 if we loaded the model in float16)
        iput_dt = input.dtype
        input = input.to(torch.float32)  # upcast input

        task_id = self.routing_infos.task_ids
        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        if self.selector is not None:
            mixing_weights = self.selector(self.routing_infos, input=input).to(
                input.device
            )
        else:
            bs = input.size(0)
            mixing_weights = torch.ones(
                bs, self.n_splits, self.n_skills, device=input.device, dtype=input.dtype
            )
        output = super(SkilledLoRA_MergeLoraAfterOP, self).forward(
            input.to(iput_dt), mixing_weights
        )
        output = output.to(iput_dt)  # downcast output
        return output

    @classmethod
    def modify_transformer(cls, transformer, config):
        config.router_selector = "vsmear_xr4"

        return modify_with_routing(cls, transformer, config, RouterWrapper)


@register_modifier("vsmear_xr1")
class VSmearXR1_AuxRoutingLoRALinear_MergeAfterOP(AuxRoutingLoRALinear_MergeAfterOP):
    """
    Like smear, but can do merging after the outer product.
    """

    @classmethod
    def modify_transformer(cls, transformer, config):
        config.router_selector = "smear"

        return modify_with_routing(cls, transformer, config, RouterWrapper)


# same as smear, but uses merging after the ouyter product
@register_modifier("smear_oracle")
class VSmearXR1_AuxRoutingLoRALinear_MergeAfterOP(AuxRoutingLoRALinear_MergeAfterOP):
    @classmethod
    def modify_transformer(cls, transformer, config):
        config.router_selector = "smear_oracle"

        return modify_with_routing(cls, transformer, config, RouterWrapper)
