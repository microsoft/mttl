import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from types import MethodType
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from mttl.models.utils import RoutingInfo
from mttl.models.cluster_reader import ClusterResult
import math

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
            setattr(object, method, MethodType(
                getattr(SkilledModel, method), object))
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
                        AverageSelector(inner_mod.n_skills,
                                        inner_mod.n_splits),
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
            print("Resizing module_logits of selector",
                  name, "with", n_tasks, "tasks.")
            selector.resize_module_logits(n_tasks)


class PolytroponAdapter(nn.Module):
    @property
    def routing_infos(self) -> RoutingInfo:
        return self.task_id_ptr["routing_infos"]


def get_selector(config):
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
    elif config.poly_selector == "cluster_hard":
        return ClusterSelector(config, soft=False)
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

        top_probs, top_indices = probs.topk(
            self.topk, dim=-1)  # 2 active skills per task
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
                module_logits = module_logits + \
                    (module_logits_disc - module_logits).detach()
            else:
                module_logits = torch.sigmoid(module_logits)

            if self.dropout > 0.0:
                module_logits = nn.Dropout(self.dropout)(module_logits)

            if self.poly_use_shared_skill:
                # last skill is always active whatever the task that has been selected
                module_logits = torch.cat((
                    module_logits[:, :, :-1], module_logits[:,
                                                            :, -1:] * 0.0 + 1.0
                ), dim=-1)

            if self.poly_average_correction:
                module_weights = module_logits * \
                    (np.sqrt(self.n_splits) / np.sqrt(self.n_skills))
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
            "module_logits", torch.empty(
                n_splits, n_skills).fill_(1.0 / n_skills)
        )

    def forward(self, routing_infos):
        bs = routing_infos.task_ids.size(0)
        module_logits = self.module_logits.view(
            1, self.n_splits, self.n_skills)
        return module_logits.expand(bs, -1, -1)


class PrivateSelector(Selector):
    def __init__(self, config):
        super().__init__()

        self.n_skills = config.n_skills

    def forward(self, routing_infos):
        return F.one_hot(routing_infos.task_ids, num_classes=self.n_skills).unsqueeze(1)


class ClusterSelector(Selector):
    def __init__(self, config, soft=False):
        super().__init__()

        self.soft = soft
        self.n_skills = config.n_skills
        self.cluster_result = ClusterResult(config.example_to_ids_path)
        self.temperature = config.poly_selector_cluster_temp

        # just to get the device and the working dtype
        self.dummy_parameter = nn.Parameter(
            torch.zeros(1), requires_grad=False)

    def forward(self, routing_infos):
        # this should return a bs x n_clusters tensor that sums to 1
        if self.soft:
            distances = self.cluster_result.get_distances_batch(
                routing_infos.hashes)
            distances = torch.tensor(
                distances,
                device=self.dummy_parameter.device,
            )
            routing = F.softmax(-distances / self.temperature,
                                dim=-1).unsqueeze(1)
        else:
            cluster_ids = torch.tensor(
                [self.cluster_result.get_cluster(h)
                 for h in routing_infos.hashes],
                device=self.dummy_parameter.device,
            )
            routing = F.one_hot(
                cluster_ids, num_classes=self.n_skills).unsqueeze(1)
        return routing.to(dtype=self.dummy_parameter.dtype)


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

        self.order = 4
        self.tensor_rank = self.n_skills
        if selector is None:
            self.selector = get_selector(config)
        else:
            self.selector = selector

        # self.lora_a = nn.Parameter(
        #     self.weight.new_empty(
        #         self.n_splits,
        #         self.n_skills,
        #         linear_layer.in_features // self.n_splits,
        #         self.rank,
        #     )
        # )

        # self.lora_b = nn.Parameter(
        #     self.weight.new_empty(
        #         self.n_splits,
        #         self.n_skills,
        #         self.rank,
        #         linear_layer.out_features // self.n_splits,
        #     )
        # )
        self.embedding_size = linear_layer.out_features
        self.embedding_dim_leaf = math.ceil(
            (self.embedding_size) ** (1 / self.order)
        )

        self.weight_leafs_a = nn.Parameter(
            self.weight.new_empty(
                self.order,
                self.tensor_rank,
                self.rank,
                self.embedding_dim_leaf,
            )
        )

        self.weight_leafs_b = nn.Parameter(
            self.weight.new_empty(
                self.order,
                self.tensor_rank,
                self.rank,
                self.embedding_dim_leaf,
            )
        )
        # we create our construction for the tensor product
        self.layerone_normalization = nn.LayerNorm(
            normalized_shape=[self.rank, self.embedding_dim_leaf**2])
        self.layertwo_normalization = nn.LayerNorm(
            normalized_shape=[self.rank, self.embedding_dim_leaf**2])
        self.reset_parameters()

        # self.lora_a = self.tensor_product_construct(self.weight_leafs_a, self.in_features).to("cuda")
        # self.lora_b = self.tensor_product_construct(self.weight_leafs_b, self.out_features).to("cuda")
        # self.lora_a = self.lora_a.transpose(2,1).unsqueeze(0)
        # self.lora_b = self.lora_b.unsqueeze(0)

    def reset_parameters(self):
        import math

        if self.kaiming_init:
            for skill in range(self.n_skills):
                for split in range(self.n_splits):
                    param = torch.empty(
                        (self.rank, self.in_features // self.n_splits))
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    self.lora_a.data[split, skill, :, :] = param.T
        else:
            gain = nn.init.calculate_gain(
                nonlinearity="leaky_relu", param=math.sqrt(5))
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
    def tensor_product_construct(self, weight_leafs, embedding_dim):
        if self.order == 4:
            w = weight_leafs
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
            # print(w[:,:,:,:].size())
            w01 = w01.view(self.tensor_rank,  self.rank, -1)
            w01 = self.layerone_normalization(w01)
            # print(w01.size())
            w23 = (w[2, :, :, :, None] * w[3, :, :, None, :])
            w23 = w23.view(self.tensor_rank,  self.rank, -1)
            w23 = self.layertwo_normalization(w23)

            w0123 = (w01[:, :, :, None] * w23[:, :, None, :])
            w0123 = w0123.view(self.tensor_rank,  self.rank, -1)
            return w0123[:, :, : embedding_dim]

    def forward(self, input):
        if self.training:
            self.training_steps += 1

        task_id = self.routing_infos.task_ids

        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        mixing_weights = self.selector(
            self.routing_infos).to(dtype=input.dtype)
        # the number of rank equals to the rank
        bs, n_splits, n_skills = mixing_weights.size()

        self.lora_a = self.tensor_product_construct(
            self.weight_leafs_a, self.in_features)  # [tensor rank, rank, D]
        self.lora_b = self.tensor_product_construct(
            self.weight_leafs_b, self.out_features)
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


class PolyLoRATensorOrder(PolytroponAdapter):
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

        # In this case, the order is exactly the number of splits.
        self.order = self.n_splits

        self.tensor_rank = self.n_skills
        if selector is None:
            self.selector = get_selector(config)
        else:
            self.selector = selector

        self.embedding_dim_leaf_a = math.ceil(
            (self.in_features) ** (1 / self.order)
        )

        self.embedding_dim_leaf_b = math.ceil(
            (self.out_features) ** (1 / self.order)
        )

        # self.embedding_dim_leaf = math.ceil(
        #     (self.out_features) ** (1 / self.order)
        # )

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

        # self.layerone_normalization = nn.LayerNorm(
        #     normalized_shape=[self.rank, self.embedding_dim_leaf**2])
        # self.layertwo_normalization = nn.LayerNorm(
        #     normalized_shape=[self.rank, self.embedding_dim_leaf**2])
        self.reset_parameters()

    def reset_parameters(self):
        import math

        if self.kaiming_init:
            for skill in range(self.n_skills):
                for split in range(self.n_splits):
                    param = torch.empty(
                        (self.rank, self.in_features // self.n_splits))
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    self.lora_a.data[split, skill, :, :] = param.T
        else:
            gain = nn.init.calculate_gain(
                nonlinearity="leaky_relu", param=math.sqrt(5))
            std = gain / math.sqrt(self.in_features)

            with torch.no_grad():
                #self.weight_leafs_a.uniform_(-std, std)
                torch.nn.init.xavier_uniform_(self.weight_leafs_a)

        # ensure that initially, adding the adapter does not change the output
        if self.use_warmup or self.lora_randb_init:
            with torch.no_grad():
                self.weight_leafs_b.uniform_(-std, std)
        else:
            #torch.nn.init.zeros_(self.weight_leafs_b)
            torch.nn.init.xavier_uniform_(self.weight_leafs_b)

    def tensor_product_construct(self, tensor_parameters, embedding_dim, flag):
        if self.order == 1:
            w = tensor_parameters.squeeze(1)
            return w
        if self.order == 2:
            w = tensor_parameters
            batch_size = w.size()[0]
            w01 = torch.einsum("bri,brj->brij", w[:, 0], w[:, 1])
            w01 = nn.LayerNorm(w01.shape[-2:]).cuda()(w01)
            w01 = w01.view(batch_size, self.rank, -1)
            w = w01[:, :, : embedding_dim]
            #w = self.layerone_normalization(w)
            return w
        if self.order == 4:
            w = tensor_parameters
            batch_size = w.size()[0]
            w01 = torch.einsum("bri,brj->brij", w[:, 0], w[:, 1])
            # w01 = w[:, 0, :, :, None] * w[:, 1, :, None, :]
            # print(w[:,:,:,:].size())
            w01 = nn.LayerNorm(w01.shape[-2:]).cuda()(w01)
            w01 = w01.view(batch_size, self.rank, -1)
            #w01 = self.layerone_normalization(w01)

            w23 = torch.einsum("bri,brj->brij", w[:, 2], w[:, 3])
            w23 = w23.view(batch_size, self.rank, -1)
            w23 = nn.LayerNorm(w23.shape[-2:]).cuda()(w23)
            #w23 = self.layertwo_normalization(w23)
            # print(w23.size())
            w0123 = (w01[:, :, :, None] * w23[:, :, None, :])
            w0123 = w0123.view(batch_size, self.rank, -1)
            #w0123 = nn.LayerNorm(w0123.shape[-2:]).cuda()(w0123)

            return w0123[:, :, : embedding_dim]

    def forward(self, input):
        if self.training:
            self.training_steps += 1

        task_id = self.routing_infos.task_ids

        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        mixing_weights = self.selector(
            self.routing_infos).to(dtype=input.dtype)
        # the number of rank equals to the rank
        bs, n_splits, n_skills = mixing_weights.size()

        # A is    order, tensor_rank, leaf
        # mixing_weights [batch_size, order, tensor_rank]
        # self.weight_leafs_a [order, tensor_rank, rank,leaf_embedding_size]
        A = torch.einsum("bos,osrl->borl",
                         (mixing_weights, self.weight_leafs_a))
        B = torch.einsum("bos,osrl->borl",
                         (mixing_weights, self.weight_leafs_b))

        A = self.tensor_product_construct(A, self.in_features, "up")  # [brd]
        A = A.transpose(2, 1)  # [bdr]
        B = self.tensor_product_construct(
            B, self.out_features, "down")  # [brd]

        adapter_out = input.bmm(A).bmm(B) / self.rank
        #print("self.weight_leafs_b", self.weight_leafs_b)
        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup

        return F.linear(input, self.weight, self.bias) + adapter_out


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
                    param = torch.empty(
                        (self.rank, self.in_features // self.n_splits))
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    self.lora_a.data[split, skill, :, :] = param.T
        else:
            gain = nn.init.calculate_gain(
                nonlinearity="leaky_relu", param=math.sqrt(5))
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

        mixing_weights = self.selector(
            self.routing_infos).to(dtype=input.dtype)
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


class PolyIA3Tensor(PolytroponAdapter):
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

        self.embedding_size = linear_layer.out_features
        self.order = 4
        self.tensor_rank = self.n_skills
        self.embedding_dim_leaf = math.ceil(
            (self.embedding_size) ** (1 / self.order)
        )
        print("leaf_dim", self.embedding_dim_leaf)
        self.weight_leafs = nn.Parameter(
            torch.ones(
                self.order,
                self.tensor_rank,
                1,
                self.embedding_dim_leaf,
            )
        )
        print("self.weight_leafs_size", self.weight_leafs.size())

        if selector is None:
            self.selector = get_selector(config)
        else:
            self.selector = selector

    def tensor_product_construct(self, weight_leafs, embedding_dim):
        if self.order == 4:
            w = weight_leafs
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
            # print(w[:,:,:,:].size())
            w01 = w01.view(self.tensor_rank, 1, -1)
            # w01 = self.layerone_normalization(w01)
            # print(w01.size())
            w23 = (w[2, :, :, :, None] * w[3, :, :, None, :])
            w23 = w23.view(self.tensor_rank,  1, -1)
            # w23 = self.layertwo_normalization(w23)

            w0123 = (w01[:, :, :, None] * w23[:, :, None, :])
            w0123 = w0123.view(self.tensor_rank,  1, -1)
            return w0123[:, :, : embedding_dim]

    def forward(self, input):
        task_id = self.routing_infos.task_ids

        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        # bs, n_splits, n_skills
        mixing_weights = self.selector(self.routing_infos)

        # construct the weight: tensor_rank, 1, embedding_size
        weight = self.tensor_product_construct(
            self.weight_leafs, self.embedding_size)

        A = torch.einsum("bqs,sqd->bqd", (mixing_weights, weight))
        A = A.reshape(input.size(0), 1, -1)
        return F.linear(input, self.weight, self.bias) * A


class PolyIA3TensorOrder(PolytroponAdapter):
    """PolyIA3 

    The rank is the number of skills and the splits is order of tensor. 

    Args:
        PolytroponAdapter (_type_): _description_
    """

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

        self.embedding_size = linear_layer.out_features
        self.order = self.n_splits
        self.tensor_rank = self.n_skills
        self.embedding_dim_leaf = math.ceil(
            (self.embedding_size) ** (1 / self.order)
        )
        print("leaf_dim", self.embedding_dim_leaf)
        self.weight_leafs = nn.Parameter(
            torch.ones(
                self.order,
                self.tensor_rank,
                1,
                self.embedding_dim_leaf,
            )
        )
        print("self.weight_leafs_size", self.weight_leafs.size())

        if selector is None:
            self.selector = get_selector(config)
        else:
            self.selector = selector

    def tensor_product_construct(self, tensor_parameters, embedding_dim):

        if self.order == 1:
            w = tensor_parameters.squeeze(1)
            return w
        if self.order == 2:
            w = tensor_parameters
            batch_size = w.size()[0]
            w01 = torch.einsum("bri,brj->brij", w[:, 0], w[:, 1])
            w01 = nn.LayerNorm(w01.shape[-2:]).cuda()(w01)
            w01 = w01.view(batch_size, self.rank, -1)
            w = w01[:, :, : embedding_dim]
            #w = self.layerone_normalization(w)
            return w
        if self.order == 4:
            w = tensor_parameters
            batch_size = w.size()[0]
            w01 = torch.einsum("bri,brj->brij", w[:, 0], w[:, 1])
            # w01 = w[:, 0, :, :, None] * w[:, 1, :, None, :]
            # print(w[:,:,:,:].size())
            w01 = nn.LayerNorm(w01.shape[-2:]).cuda()(w01)
            w01 = w01.view(batch_size, self.rank, -1)
            #w01 = self.layerone_normalization(w01)

            w23 = torch.einsum("bri,brj->brij", w[:, 2], w[:, 3])
            w23 = w23.view(batch_size, self.rank, -1)
            w23 = nn.LayerNorm(w23.shape[-2:]).cuda()(w23)
            #w23 = self.layertwo_normalization(w23)
            # print(w23.size())
            w0123 = (w01[:, :, :, None] * w23[:, :, None, :])
            w0123 = w0123.view(batch_size, self.rank, -1)
            w0123 = nn.LayerNorm(w0123.shape[-2:]).cuda()(w0123)

            return w0123[:, :, : embedding_dim]

    def forward(self, input):
        task_id = self.routing_infos.task_ids

        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        # bs, order, rank
        mixing_weights = self.selector(self.routing_infos)
        A = torch.einsum("bor,ored->bored",
                         (mixing_weights, self.weight_leafs))

        # construct the weight: tensor_rank, 1, embedding_size
        aggregation = self.tensor_product_construct(A, self.embedding_size)
        # A = A.reshape(input.size(0), 1, -1)
        return F.linear(input, self.weight, self.bias) * aggregation

    def extra_repr(self):
        return "n_skills={}, in_features={}, out_features={}, bias={}".format(
            self.n_skills, self.in_features, self.out_features, self.bias is not None
        )


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
            'coarsegrained': None,
            'finegrained': None,
            'layerwise': 'layer',
            'blockwise': 'block',
            'coderwise': 'coder'
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
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."

                    identifier = _extract_identifier(
                        f'{m_name}.{c_name}', config.poly_granularity)
                    if identifier not in selectors.keys():
                        selectors[identifier] = get_selector(config)
                    selector = selectors[identifier]
                    total_layers += 1

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
        f'created {len(selectors)} for a total of {total_layers} adapted layers')
    return SkilledModel.register_functions(transformer)


def modify_with_tensororderpoly_ia3(transformer, config):
    return modify_with_poly(transformer, config, PolyIA3TensorOrder)


def modify_with_tensorpoly_ia3(transformer, config):
    return modify_with_poly(transformer, config, PolyIA3Tensor)


def modify_with_poly_ia3(transformer, config):
    return modify_with_poly(transformer, config, PolyIA3Linear)


def modify_with_poly_lora(transformer, config):
    return modify_with_poly(transformer, config, PolyLoRALinear)


def modify_with_tensorpoly_lora(transformer, config):
    return modify_with_poly(transformer, config, PolyLoRATensor)


def modify_with_tensororderpoly_lora(transformer, config):
    return modify_with_poly(transformer, config, PolyLoRATensorOrder)
