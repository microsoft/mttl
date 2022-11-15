import math
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn.init import calculate_gain

EPS = 1e-12


class Adapter(nn.Module):
    def __init__(self, args, old_module, selector=None):
        super().__init__()

        if selector is None:
            selector = get_selector(args.selector)(args)

        self.selector = selector
        self.processor = get_processor(args.processor)(args, old_module)

        if args.combinor == args.processor:
            self.combinor = self.processor
        else:
            self.combinor = get_combinor(args.combinor)(args)

    def forward(self, input, task=None, *args, **kwargs):
        """do your regular adapter forward pass here"""

        params = self.processor.get_params()
        module_weights = self.selector.select_modules()
        combined_modules = self.combinor.combine_modules(module_weights, params)
        output = self.processor(input, params=combined_modules)
        return output


class Selector(ABC, nn.Module):
    @abstractmethod
    def select_modules(self) -> torch.Tensor:
        pass


class Polytropon(Selector):
    """Return a distribution over modules for each task"""

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.n_skills = args.n_skills
        self.module_logits = nn.Parameter(self.init_module_logits(self.args.n_tasks))
        self._reserved_skill = False

    def init_module_logits(self, n_tasks):
        return torch.empty((n_tasks, self.args.n_splits * self.args.n_skills)).uniform_(
            -1e-3, 1e-3
        )

    def select_modules(self) -> torch.Tensor:
        # `inform_layers` should have been called
        assert self.tasks is not None

        # now sample things accordingly (during training)
        if self.training:
            module_probs = RelaxedBernoulli(
                temperature=1.0, logits=self.module_logits[self.tasks]
            ).rsample()
        else:
            # test time sigmoid
            module_probs = torch.sigmoid(self.module_logits[self.tasks])

        module_probs = module_probs.view(-1, self.args.n_splits, self.n_skills)
        self.module_weights = module_probs / (
            module_probs.sum(dim=-1, keepdim=True) + EPS
        )
        return self.module_weights


class Private(Selector):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.n_skills == args.n_tasks, "need 1 skill for each task"

    def select_modules(self) -> torch.Tensor:
        assert self.tasks is not None
        return F.one_hot(self.tasks, num_classes=self.args.n_skills)


class Shared(Selector):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def select_modules(self) -> torch.Tensor:
        assert self.tasks is not None

        n_skills = self.args.n_skills
        assert self.args.n_splits == 1, "only 1 split supported for shared selector"

        out = self.tasks.new_empty(self.tasks.size(0), 1, n_skills, dtype=torch.float)
        out.fill_(1.0 / n_skills)
        return out


class Average(Selector):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def select_modules(self) -> torch.Tensor:
        assert self.tasks is not None

        out = self.tasks.new_empty(
            self.tasks.size(0),
            self.args.n_splits,
            self.args.n_skills,
            dtype=torch.float,
        )
        out.fill_(1.0 / self.args.n_skills)
        return out


class Custom(Selector):
    def __init__(self, args):
        raise NotImplementedError()


def get_selector(name):
    return {
        "none": None,
        "polytropon": Polytropon,
        "average": Average,
        "shared": Shared,
        "private": Private,
    }[name]


class Combinor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def combine_modules(self, probs, param_dict, *args, **kwargs):
        pass


class WeightedSum(Combinor):
    def combine_modules(self, probs, param_dict, *args, **kwargs):
        bs, n_splits, n_skills = probs.shape
        out = OrderedDict()

        for name, param in param_dict.items():
            assert param.size(0) == n_splits and param.size(1) == n_skills
            if len(param.shape) == 4:
                p_mod = torch.einsum("bpt,ptrd->bprd", probs, param)
            else:
                p_mod = torch.einsum("bpt,ptd->bpd", probs, param)
            out[name] = p_mod
        return out


class NormedSum(Combinor):
    def combine_modules(self, probs, param_dict, *args, **kwargs):
        bs, n_splits, n_skills = probs.shape
        out = OrderedDict()

        probs = probs / ((probs ** 2.0).sum(-1) ** 0.5)[:, :, None]
        for name, param in param_dict.items():
            assert param.size(0) == n_splits and param.size(1) == n_skills
            if len(param.shape) == 4:
                p_mod = torch.einsum("bpt,ptrd->bprd", probs, param)
            else:
                p_mod = torch.einsum("bpt,ptd->bpd", probs, param)
            out[name] = p_mod
        return out


def get_combinor(name):
    return {
        "weighted_sum": WeightedSum,
        "normed_sum": NormedSum,
    }[name]


class LoRA(nn.Module):
    def __init__(self, args, old_linear):
        super().__init__()

        self.args = args
        self.r = args.lora_rank
        self.out_features, self.in_features = old_linear.weight.shape

        # For now, always keep old layer as frozen
        self.old_linear = old_linear
        self.old_linear.requires_grad = False

        assert self.in_features % self.args.n_splits == 0
        assert self.out_features % self.args.n_splits == 0

        weight_A = old_linear.weight.new_empty(
            args.n_splits, args.n_skills, self.in_features // self.args.n_splits, self.r
        )
        weight_B = old_linear.weight.new_empty(
            args.n_splits,
            args.n_skills,
            self.r,
            self.out_features // self.args.n_splits,
        )

        self.params = nn.ParameterDict(
            {
                "A": nn.Parameter(weight_A),
                "B": nn.Parameter(weight_B),
            }
        )

        self.scaling = 1 / self.r
        self.reset_parameters()

    def get_params(self):
        return self.params

    def reset_parameters(self):
        gain = calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.params["A"].uniform_(-std, std)

        # ensure that initially, adding the adapter does not change the output
        torch.nn.init.zeros_(self.params["B"])

    def forward(self, input, params=None):
        bs, sq, D = input.size()

        if params is None:
            params = self.params

        A, B = params["A"], params["B"]

        # might need to repeat weights when e.g. doing beam search
        repeat = bs // A.size(0)

        if repeat > 1:
            A = A.repeat_interleave(repeat, dim=0)
            B = B.repeat_interleave(repeat, dim=0)

        # --- Remove the `n_splits` axis
        # (bs, n_splits, D // n_splits, rank) and (bs, n_splits, rank, D // n_splits)
        A = A.reshape(bs, -1, self.r)
        B = B.transpose(1, 2).reshape(bs, self.r, -1)

        output = torch.matmul(input, A)
        output = torch.matmul(output, B)

        return self.old_linear(input) + output * self.scaling


class IA3(nn.Module):
    def __init__(self, args, old_linear):
        super().__init__()

        self.out_features, self.in_features = old_linear.weight.shape

        # For now, always keep old layer as frozen
        self.old_linear = old_linear
        self.old_linear.requires_grad = False

        weight_B = old_linear.weight.new_empty(
            args.n_splits, args.n_skills, self.out_features // args.n_splits
        )

        self.params = nn.ParameterDict(
            {
                "B": nn.Parameter(weight_B),
            }
        )
        self.reset_parameters()

    def get_params(self):
        return self.params

    def reset_parameters(self):
        # ensure that initially, adding the adapter does not change the output
        torch.nn.init.ones_(self.params["B"])

    def forward(self, input, params=None):
        bs, sq, D = input.size()

        if params is None:
            params = self.params

        # might need to repeat weights when e.g. doing beam search
        repeat = bs // params["B"].size(0)

        if repeat > 1:
            params["B"] = params["B"].repeat_interleave(repeat, dim=0)

        # flatten the n_splits dim
        # bs, n_skills, D // n_skills --> bs, 1, D
        params["B"] = params["B"].view(bs, 1, self.out_features)

        output = self.old_linear(input) * params["B"]
        return output


def get_processor(name):
    return {
        "none": None,
        "lora": LoRA,
        "ia3": IA3,
    }[name]
