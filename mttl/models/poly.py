import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from types import MethodType
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from mttl.models.utils import RoutingInfo


EPS = 1e-12


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
    def switch_selector_to_average(object):
        """Switches PolytroponSelector to AverageSelector.
        """
        for name, module in object.named_modules():
            for name, inner_mod in module.named_children():
                if isinstance(inner_mod, PolytroponSelector):
                    print(
                        "Replacing with average: ",
                        name,
                        "n_skills:",
                        inner_mod.n_skills,
                    )
                    setattr(
                        module,
                        name,
                        AverageSelector(inner_mod.n_skills, inner_mod.n_splits),
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


class PolytroponAdapter(nn.Module):
    @property
    def routing_infos(self) -> RoutingInfo:
        return self.task_id_ptr["routing_infos"]


def get_selector(config):
    if config.poly_selector == "poly":
        return PolytroponSelector(config)
    elif config.poly_selector == "private":
        return PrivateSelector(config)
    elif config.poly_selector == "moe":
        return MoESelector(config)
    else:
        raise NotImplementedError()


class Selector(nn.Module):
    pass


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


class PolyLoRALinear(PolytroponAdapter):
    def __init__(self, config, task_id_ptr, linear_layer, selector=None):
        super().__init__()
        self.n_splits = config.n_splits
        self.n_tasks = config.n_tasks
        self.n_skills = config.n_skills
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

        if selector is None:
            self.selector = get_selector(config)
        else:
            self.selector = selector

        self.lora_a = nn.Parameter(
            self.weight.new_empty(
                self.n_splits,
                self.n_skills,
                linear_layer.in_features // self.n_splits,
                self.rank,
            )
        )
        self.lora_b = nn.Parameter(
            self.weight.new_empty(
                self.n_splits,
                self.n_skills,
                self.rank,
                linear_layer.out_features // self.n_splits,
            )
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
                self.lora_a.uniform_(-std, std)


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

        mixing_weights = self.selector(self.routing_infos).to(dtype=input.dtype)
        bs, n_splits, n_skills = mixing_weights.size()

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

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    identifier = _extract_identifier(f'{m_name}.{c_name}', config.poly_granularity)
                    if identifier not in selectors.keys():
                        selectors[identifier] = get_selector(config)
                    selector = selectors[identifier]
                    total_layers += 1

                    print(f"Patching {m_name}.{c_name}...")
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

    print(f'created {len(selectors)} selectors for a total of {total_layers} adapted layers')
    return SkilledModel.register_functions(transformer)


def modify_with_poly_ia3(transformer, config):
    return modify_with_poly(transformer, config, PolyIA3Linear)


def modify_with_poly_lora(transformer, config):
    return modify_with_poly(transformer, config, PolyLoRALinear)
