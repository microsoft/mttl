from torch import nn

from mttl.models.containers.lora_containers import ExpertContainer
from mttl.models.containers.selectors.base import KVTaskNameSelector
from mttl.models.library.expert import Expert
from mttl.models.modifiers.kv_adapter import KVAdapter, KVAdapterConfig


class KVExpertContainer(KVAdapter, ExpertContainer):
    """Expert Container for KVAdapters.
    Unlike the LoRAExpertContainer, the KVExpertContainer is a KVAdapter itself,

    See `KVSelector` for info on how the routing is done.
    See `KVAdapter` for info on the control flow of the forward pass.
    """

    __supports_configs__ = [KVAdapterConfig]

    def __init__(self, config, layer, selector=None, **kwargs):
        super().__init__(
            config,
            layer,
            selector or KVTaskNameSelector(),
        )

        # Check if layer is an attention layer :
        if not hasattr(self.attn_layer, "k_proj") and self.config.model != "phi-2":
            raise ValueError(
                "`KVExpertContainer` should wrap an attention layer. {}".format(
                    self.attn_layer.__class__.__name__
                )
            )

        self.experts = nn.ModuleDict({})

    # skip creating the adapter weights
    def create_for_layer(self, attn_layer):
        pass

    # Delegate Routing ops to the selectors
    def route(self, query, keys, attn_layer=None):
        if callable(getattr(self.selector, "route", None)):
            return self.selector.route(self.experts, query, keys, attn_layer)

        # This behavior is problematic! you need `get_gate` to call the adapter method
        return super().route(query, keys, attn_layer)

    # Delegate Routing ops to the selectors
    def aggregate(self, adapter_weights, adapter_v):
        if callable(getattr(self.selector, "aggregate", None)):
            return self.selector.aggregate(self.experts, adapter_weights, adapter_v)

        # This behavior is problematic! you need `get_gate` to call the adapter method
        return super().aggregate(adapter_weights, adapter_v)

    def __getitem__(self, key):
        raise NotImplementedError()

    def __len__(self):
        return len(self.expert_names)

    def get_kv_weights(self, k_proj, v_proj):
        return self.selector.get_kv_weights(self.experts, k_proj, v_proj)

    def get_gate(self, adapter_weights):
        return self.selector.get_gate(self.experts, adapter_weights)

    def on_add_expert(
        self,
        expert: Expert,
        action="route",
        is_default=False,
        **kwargs,
    ) -> None:
        from mttl.models.containers import filter_expert_weights

        expert_weights = filter_expert_weights(
            self.__layer_name__, expert.expert_weights
        )

        if action == "merge":
            raise ValueError("Merging is not supported for `KVAdapters`.")

        self._check_config(expert.expert_config)

        expert_module = KVAdapter(expert.expert_config, self.attn_layer)
        expert_module.load_adapter_weights(expert_weights)
