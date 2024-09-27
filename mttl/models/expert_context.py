import functools
import threading
from dataclasses import dataclass, field, fields
from typing import Dict, List

import torch


@dataclass
class RoutingInfo:
    input_ids: torch.Tensor = None
    labels: torch.Tensor = None
    attention_mask: torch.Tensor = None
    task_ids: torch.Tensor = None
    task_names: List[str] = None
    task_sources: List[str] = None
    example_ids: List[int] = None
    sources_texts: List[str] = None
    labels_texts: List[str] = None
    task_weights: torch.nn.ParameterDict = None
    aux_losses: Dict = field(default_factory=dict)
    packed_seq_lens: List[int] = None
    seq_lens: List[int] = None
    packed_attn_mask: torch.Tensor = None
    # skill_mixing_coefs: torch.Tensor = None
    # modality_names: List[str] = None

    @classmethod
    def pop_elements(cls, batch, keep=None):
        """We don't want to pass these elements to the model."""
        keep = keep or []
        return [
            batch.pop(k.name)
            for k in fields(cls)
            if k.name in batch and k.name not in keep
        ]

    @classmethod
    def prepare_for_forward(cls, batch):
        cls.pop_elements(batch, keep=["input_ids", "attention_mask", "labels"])

    @classmethod
    def prepare_for_generate(cls, batch):
        cls.pop_elements(batch, keep=["input_ids", "attention_mask"])

    @classmethod
    def from_batch(cls, batch: dict, **kwargs):
        task_ids = batch.get("task_ids").long() if "task_ids" in batch else None
        task_names = batch.get("task_names", None)
        task_weights = batch.get("task_weights", None)
        task_sources = batch.get("task_sources", None)

        ri = cls(
            task_ids=task_ids,
            task_names=task_names,
            task_weights=task_weights,
            task_sources=task_sources,
            input_ids=batch.get("input_ids", None),
            example_ids=batch.get("example_ids", None),
            sources_texts=batch.get("sources_texts", None),
            labels=batch.get("labels", None),
            attention_mask=batch.get("attention_mask", None),
            packed_seq_lens=batch.get("packed_seq_lens", None),
            seq_lens=batch.get("seq_lens", None),
            packed_attn_mask=batch.get("packed_attn_mask", None),
            **kwargs,
        )
        return ri

    def _repeat(self, inputs, n):
        if inputs is not None:
            if isinstance(inputs, torch.Tensor):
                return inputs.repeat_interleave(n)
            else:
                return [item for item in inputs for _ in range(n)]
        return inputs

    def repeat_interleave(self, repeats):
        # useful for beam search
        self.task_ids = self._repeat(self.task_ids, repeats)
        self.task_names = self._repeat(self.task_names, repeats)
        self.task_sources = self._repeat(self.task_sources, repeats)
        self.example_ids = self._repeat(self.example_ids, repeats)
        self.task_weights = self._repeat(self.task_weights, repeats)


class InfoContainer:
    local = threading.local()
    local.context = None
    routing_info_cls = RoutingInfo

    def __init__(self, model, routing_infos=None, **kwargs):
        self.model = model
        # stores the routing info for the model
        self._routing_infos = routing_infos
        # stores the routing gates for each layer, if any
        self._routing_gates = []

    def __enter__(self):
        InfoContainer.local.context = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # delete the context
        InfoContainer.local.context = None

    @classmethod
    def get(cls):
        return cls.local.context

    @property
    def routing_infos(self):
        return self._routing_infos

    @property
    def routing_gates(self):
        return self._routing_gates

    @routing_infos.setter
    def routing_infos(self, value: "RoutingInfo"):
        self._routing_infos = value

    @routing_gates.setter
    def routing_gates(self, value: List):
        self._routing_gates = value

    @classmethod
    def register_routing_info_class(cls, routing_info_cls):
        cls.routing_info_cls = routing_info_cls

    @classmethod
    def create_context(cls, f):
        """
        Decorator method that wraps a ``forward`` or ``generate`` function of a model class.
        """

        if f.__name__ not in ["forward", "generate"]:
            raise ValueError(
                f"Unknown wrap method: {f.__name__}, must be 'forward' or 'generate'."
            )

        @functools.wraps(f)
        def wrapper_func(model, **kwargs):
            if "input_ids" not in kwargs:
                raise ValueError(
                    "The first argument of the function to wrap must be 'input_ids'."
                )

            return_context = kwargs.pop("return_context", False)
            with cls(model, cls.routing_info_cls.from_batch(kwargs)) as context:
                if f.__name__ == "forward":
                    cls.routing_info_cls.prepare_for_forward(kwargs)
                elif f.__name__ == "generate":
                    cls.routing_info_cls.prepare_for_generate(kwargs)

                results = f(model, **kwargs)
                if return_context:
                    context_returns = {
                        "routing_infos": context.routing_infos,
                        "routing_gates": context.routing_gates,
                    }
                    return results, context_returns
            return results

        return wrapper_func

    @classmethod
    def wrap_with_context(cls, f):
        """
         Decorator method that wraps a general function of a model class
        (We may want to wrap other methods than just forward and generate).
        Use `create_context` whenever possible
        """

        @functools.wraps(f)
        def wrapper_func(model, *args, **kwargs):

            return_context = kwargs.pop("return_context", False)
            with cls(model, cls.routing_info_cls.from_batch(args[0])) as context:
                results = f(model, *args, **kwargs)
                if return_context:
                    context_returns = {
                        "routing_infos": context.routing_infos,
                        "routing_gates": context.routing_gates,
                    }
                    return results, context_returns
            return results

        return wrapper_func
