from typing import Optional, Any, Dict
from torch import nn
import torch
import math
from torch.autograd import Function
from torch.nn.modules.module import Module
import bitsandbytes as bnb
from abc import ABC, abstractmethod
from typing import Dict

expert_ids2name = {}


class Adapter(nn.Module):
    @property
    def layer_name(self):
        if not hasattr(self, "__layer_name__"):
            raise ValueError(
                "Layer name not set, dependency injection not done properly?"
            )

        return self.__layer_name__


class MergableAdapter(ABC):
    @abstractmethod
    def merge_with_layer(self):
        pass


class LoRA(Adapter, MergableAdapter):
    def __init__(
        self,
        config,
        layer,
    ):
        super().__init__()

        # assign self variables
        self.config = config
        self.rank = config.lora_rank
        self.alpha = config.lora_alpha
        self.dropout = config.lora_dropout
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.init_b_random = config.lora_init_b_random
        self.training_steps = 0.0
        self.scaling = self.alpha / self.rank
        self.forward_fn = None
        self.layer = layer

        if hasattr(layer, "weight"):
            self.weight = layer.weight

        if hasattr(layer, "bias"):
            self.bias = layer.bias

        self.create_for_layer(layer)
        self.reset_parameters()
        self.merged_with_layer = False

    def load_lora_weights(self, state_dict):
        self.lora_a.data.copy_(state_dict["lora_a"])
        self.lora_b.data.copy_(state_dict["lora_b"])

    def merge_with_layer(self):
        """Merge this adapter with the layer!"""
        if isinstance(self.layer, nn.Linear):
            self.merged_with_layer = True
            # for back-compatibility, try the two sides:
            if self.lora_a.data.shape[0] == self.layer.weight.shape[0]:
                to_merge = self.lora_a.data @ self.lora_b.data
            else:
                to_merge = (self.lora_a.data @ self.lora_b.data).T
            to_merge = to_merge * self.scaling

            if isinstance(self.layer, bnb.nn.Linear8bitLt):
                if self.layer.state.SCB is None:
                    self.layer.state.SCB = self.layer.weight.SCB

                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                im = (
                    torch.eye(self.layer.weight.data.shape[-1])
                    .contiguous()
                    .half()
                    .to(self.weight.device)
                )
                im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
                im, Sim = bnb.functional.transform(im, "col32")

                if self.layer.state.CxB is None:
                    (
                        self.layer.state.CxB,
                        self.layer.state.SB,
                    ) = bnb.functional.transform(
                        self.layer.weight.data, to_order=self.layer.state.formatB
                    )

                out32, Sout32 = bnb.functional.igemmlt(
                    im, self.layer.state.CxB, Sim, self.layer.state.SB
                )
                output = bnb.functional.mm_dequant(
                    out32, Sout32, SCim, self.layer.state.SCB, bias=None
                ).t()
                w_data = output.to(to_merge.dtype).to(to_merge.device) + to_merge

                self.layer.weight = bnb.nn.Int8Params(
                    w_data.to("cpu"),
                    requires_grad=False,
                    has_fp16_weights=self.layer.weight.has_fp16_weights,
                ).to(self.layer.weight.device)
                self.layer.state.reset_grads()
            else:
                self.layer.weight.data.add_(to_merge.to(self.layer.weight.device))
        else:
            raise NotImplementedError("LoRA only supports nn.Linear layers.")

    def create_for_layer(self, layer):
        if isinstance(layer, nn.Linear):
            self.lora_a = nn.Parameter(
                torch.empty(layer.in_features, self.rank).to(
                    device=layer.weight.device
                ),
            )
            self.lora_b = nn.Parameter(
                torch.empty(self.rank, layer.out_features).to(
                    device=layer.weight.device
                ),
            )
            self.forward_fn = self.forward_linear_
        else:
            raise NotImplementedError("LoRA only supports nn.Linear layers.")

    def forward_linear_(self, input, **kwargs):
        output = self.layer(input)
        if self.merged_with_layer:
            return output
        else:
            input_lora = input.to(self.lora_a.dtype)
            adapter_out = (
                torch.matmul(torch.matmul(input_lora, self.lora_a), self.lora_b)
                * self.scaling
            )
            return output + adapter_out.to(input.dtype)

    @classmethod
    def parallel_linear_forward(cls, input, loras):
        if any([lora.merged_with_layer for lora in loras]):
            raise ValueError("Cannot parallelize merged loras.")
        if len(set([lora.layer for lora in loras])) > 1:
            raise ValueError("Cannot parallelize loras applied to different layers.")

        # (n_examples, in_features, rank)
        lora_a = torch.stack([lora.lora_a for lora in loras], dim=0)
        # (n_examples, rank, out_features)
        lora_b = torch.stack([lora.lora_b for lora in loras], dim=0)
        # (n_examples,)
        scaling = torch.cat(
            [torch.FloatTensor([lora.scaling]) for lora in loras], dim=0
        ).to(device=lora_a.device)

        # (n_examples, seq_len, out_features)
        layer_out = loras[0].layer(input)
        input_lora = input.to(loras[0].lora_a.dtype)

        adapter_out = (
            torch.bmm(torch.bmm(input_lora, lora_a), lora_b) * scaling[:, None, None]
        )
        return layer_out + adapter_out.to(dtype=input.dtype)

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

        # ensure that initially, adding the adapter does not change the output
        if self.init_b_random:
            with torch.no_grad():
                self.lora_b.uniform_(-std, std)
        else:
            torch.nn.init.zeros_(self.lora_b)

    def forward(self, *args, **kwargs):
        return self.forward_fn(*args, **kwargs)


class IA3(Adapter):
    def __init__(self, config, layer):
        super().__init__()

        assert isinstance(
            layer, nn.Linear
        ), f"IA3 can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."

        self.layer = layer
        self.multi_lora_b = nn.Parameter(
            torch.ones(layer.out_features).to(device=self.layer.weight.device)
        )

    def forward(self, input):
        return self.layer(input) * self.multi_lora_b


class LN(Adapter):
    def __init__(self, config, layer):
        super().__init__()

        self.out_features = layer.weight.size(0)
        self.weight = layer.weight
        self.variance_epsilon = layer.variance_epsilon

        assert self.out_features % config.n_splits == 0

        self.lora_b = nn.Parameter(self.weight.data)

    def forward(self, input):
        # layer norm should always be calculated in float32
        variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
        input = input / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            input = input.to(torch.float16)
        return self.lora_b.unsqueeze(0) * input


class SkilledLoRA(LoRA):
    def __init__(
        self,
        config,
        layer,
    ):
        self.n_splits = config.n_splits
        self.n_skills = config.n_skills
        super().__init__(config, layer)

    def create_for_layer(self, layer):
        if isinstance(layer, nn.Linear):
            self.lora_a = nn.Parameter(
                torch.empty(
                    self.n_skills,
                    self.n_splits,
                    layer.in_features // self.n_splits,
                    self.rank,
                ).to(device=self.weight.device)
            )
            self.lora_b = nn.Parameter(
                torch.empty(
                    self.n_skills,
                    self.rank,
                    self.n_splits,
                    layer.out_features // self.n_splits,
                ).to(device=self.weight.device)
            )
            self.forward_fn = self.forward_linear_
        else:
            raise NotImplementedError("SkilledLoRA only supports nn.Linear layers.")

    def forward_linear_(self, input, weights):
        layer_out = self.layer(input)
        input_lora = input.to(self.lora_a.dtype)
        bs = input.size(0)

        # these are task ids
        if weights.ndim == 1:
            # use indexing!
            wrm_steps = 0
            if self.training_steps < wrm_steps:
                A = self.lora_a[torch.zeros_like(weights).long()]
                B = self.lora_b[torch.zeros_like(weights).long()]
            else:
                if self.training_steps == wrm_steps:
                    self.lora_a.data.copy_(
                        self.lora_a.data[:1].repeat(self.n_skills, 1, 1, 1)
                    )
                    self.lora_b.data.copy_(
                        self.lora_b.data[:1].repeat(self.n_skills, 1, 1, 1)
                    )
                A = self.lora_a[weights.long(), :, :, :]
                B = self.lora_b[weights.long(), :, :, :]
        else:
            A = torch.einsum("bqs,sqdr->bqdr", (weights, self.lora_a))
            B = torch.einsum("bqs,srqd->brqd", (weights, self.lora_b))

        A = A.view(bs, self.in_features, self.rank)
        B = B.view(bs, self.rank, self.out_features)
        adapter_out = input_lora.bmm(A).bmm(B) * self.scaling
        return layer_out + adapter_out.to(input.dtype)

    @classmethod
    def weighted_merge_forward(cls, input, loras, weights, merge_after=False):
        """
        Meging loras into one loa accroding to the weights
        """
        if any([lora.merged_with_layer for lora in loras]):
            raise ValueError("Cannot parallelize merged loras.")
        if len(set([lora.layer for lora in loras])) > 1:
            raise ValueError("Cannot parallelize loras applied to different layers.")

        # (n_skills, in_features, rank)
        lora_a = torch.stack([lora.lora_a for lora in loras], dim=0)
        # (n_skills, rank, out_features)
        lora_b = torch.stack([lora.lora_b for lora in loras], dim=0)
        # (n_examples,)
        scaling = torch.cat(
            [torch.FloatTensor([lora.scaling]) for lora in loras], dim=0
        ).to(device=lora_a.device)
        weights = torch.stack([weight.squeeze() for weight in weights], dim=0)
        layer_out = loras[0].layer(input)
        # make sure input is in the same dtype as lora. By default it is likley to be in float16 (if we lad LLama model in float16). Butloas here can be in float32
        input = input.to(dtype=lora_a.dtype)
        if merge_after:
            adapter_out_a = torch.einsum("bsd,kdr->bskr", (input, lora_a))
            adapter_out_b = torch.einsum("bskr,krd->bskd", (adapter_out_a, lora_b))
            adapter_out = (
                torch.einsum("bskd,k->bsd", (adapter_out_b, weights)) * scaling.mean()
            )
        else:
            # merge lors according to the weights, As and Bs seperately
            A = torch.einsum("s,sdr->dr", (weights, lora_a))
            B = torch.einsum("s,srd->rd", (weights, lora_b))

            # (n_examples, seq_len, out_features)
            adapter_out = torch.matmul(torch.matmul(input, A), B) * scaling.mean()
        return layer_out + adapter_out.to(dtype=layer_out.dtype)


class ExpertContainer(Adapter, MergableAdapter):
    def __init__(self, config, task_id_container, layer, selector=None):
        super().__init__()
        self.config = config
        self.layer = layer
        self.selector = selector

        if not isinstance(self.layer, nn.Linear):
            raise ValueError(
                "Expert containers for layers other than nn.Linear have not been implemented."
            )

        self.info_container = task_id_container
        self.default_expert_name = None
        self.merged_expert_names = []
        self.experts = nn.ModuleDict({})

    def add_expert(
        self,
        name: str,
        expert_config: Any,
        expert_weights: Dict[str, torch.Tensor],
        action="merge",
        is_default=False,
    ) -> None:
        if name in self.experts:
            raise ValueError("An expert with name {} already exists.".format(name))

        if is_default and action == "merge":
            raise ValueError(
                "Cannot set is_default if this expert is merged, change to 'route'."
            )

        # hack this for now, but build a proper config for each module
        if expert_config.model_modifier == "lora":
            expert_module = LoRA(expert_config, self.layer)
            expert_module.load_lora_weights(expert_weights)
        else:
            raise NotImplementedError("ExpertContainer only supports LoRA experts.")

        if action == "merge":
            # weight is merged with layer so we can discard it now
            if expert_config.model_modifier == "lora":
                expert_module.merge_with_layer()
                self.merged_expert_names.append(name)
            else:
                raise NotImplementedError("Merging experts only supports LoRA experts.")
        else:
            # we keep track of the expert weights
            self.experts[name] = expert_module
        if is_default:
            self.default_expert_name = name

    def merge_with_layer(self):
        if len(self.experts) > 0:
            for name, expert_module in self.experts.items():
                assert isinstance(
                    expert_module, LoRA
                ), "Only LoRA experts can be merged with the layer for now."
                expert_module.merge_with_layer()
                self.merged_expert_names.append(name)
                self.experts.pop(name)

    def weighted_route(self, input, task_weights):
        """
        Route all examples according to the weights given in the weights dictionary: expert_name -> weight
        """
        load_experts = []
        weights = []

        for task_name, weight in task_weights.items():
            load_experts.append(self.experts[task_name])
            weights.append(weight)
        # assume all experts are loras
        output = SkilledLoRA.weighted_merge_forward(
            input, load_experts, weights, merge_after=True
        )

        return output

    def route_with_expert_name(self, input, expert_names):
        """
        Route according to the expert name information
        """
        load_experts = []
        for expert_name in expert_names:
            if expert_name not in self.experts:
                if not self.default_expert_name:
                    raise ValueError(
                        "The expert {} not in the expert factory. Consider setting a default expert!".format(
                            expert_name
                        )
                    )
                else:
                    selected_expert = self.default_expert_name
            else:
                selected_expert = expert_name
                load_experts.append(self.experts[selected_expert])
        output = LoRA.parallel_linear_forward(input, load_experts)
        return output

    def route_with_task_name(self, input, task_names):
        """
        Route according to the task name information
        """
        load_experts = []

        for task_name in task_names:
            if task_name not in self.experts:
                if not self.default_expert_name:
                    raise ValueError(
                        "The expert for this task {} does not exists. Consider setting a default expert!".format(
                            task_name
                        )
                    )
                else:
                    selected_expert = self.default_expert_name
            else:
                selected_expert = task_name
            load_experts.append(self.experts[selected_expert])
        # assume all experts are loras
        output = LoRA.parallel_linear_forward(input, load_experts)
        return output

    def forward(self, input, **kwargs):
        task_names = self.info_container["routing_infos"].task_names
        expert_ids = self.info_container["routing_infos"].expert_ids
        expert_names = [
            expert_ids2name.get(id, self.default_expert_name) for id in expert_ids
        ]

        if task_names and (
            any(task_name not in self.experts for task_name in task_names)
            and not self.default_expert_name
            and len(self.experts)
        ):
            raise ValueError(
                "Experts for all tasks have not been loaded! Set a default expert?"
            )

        # if it has some routing experts *and* task names, then we can route
        if len(self.experts) and self.selector is None:
            assert (
                task_names is not None
            ), "Task names are not given: set router or merge experts into the layer."
            if expert_names is not None:
                output = self.route_with_expert_name(input, expert_names)
            else:
                output = self.route_with_task_name(input, task_names)
        elif len(self.experts) and self.selector is not None:
            weights: Dict = self.selector(input)
            output = self.weighted_route(input, weights)
        else:
            ###############################################################
            ## no experts -- no routing, experts were merged into the layer
            output = self.layer(input, **kwargs)
        return output


class SkilledLoRA_MergeLoraAfterOP(SkilledLoRA):
    def __init__(
        self,
        config,
        layer,
    ):
        super().__init__(config, layer)
        self.merge_after_op = config.merge_after_op

    def forward_linear_(self, input, weights):
        if not self.merge_after_op:
            return super().forward_linear_(input, weights)

        layer_out = self.layer(input)
        input_lora = input.to(self.lora_a.dtype)

        bs, _, _ = weights.size()
        adapter_out = torch.einsum(
            "bsd,qkdr->bsqkr", (input_lora, self.lora_a)
        )  # bs seq x n_splits x n_skills x rank
        adapter_out = torch.einsum(
            "bsqkr,qkrd->bsqkd", (adapter_out, self.lora_b)
        )  # bs x seq x n_splits x n_skills x D
        adapter_out = torch.einsum(
            "bsqkd,bqk->bsd", (adapter_out, weights)
        )  # bs x seq x n_splits x D
        adapter_out *= self.scaling

        return layer_out + adapter_out.to(input.dtype)
