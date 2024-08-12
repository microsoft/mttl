import functools
import threading
from typing import List


class InfoContainer:
    local = threading.local()
    local.context = None

    def __init__(self, model, routing_infos=None, **kwargs):
        self.model = model
        # stores the routing info for the model
        self._routing_infos = routing_infos
        # stores the routing gates for each layer, if any
        self._routing_gates = []

    def __enter__(self):
        InfoContainer.local.context = self

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
    def wrap_forward(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a model class.
        """
        from mttl.models.modifiers.routing import RoutingInfo

        @functools.wraps(f)
        def wrapper_func(model, *args, **kwargs):
            if not isinstance(args[0], dict) and "input_ids" not in args[0]:
                raise ValueError(
                    "The first argument of the function to wrap must be a dictionary with 'input_ids' key."
                )

            with cls(model, RoutingInfo.from_batch(args[0])) as context:
                results = f(model, *args, **kwargs)
                if kwargs.get("return_context", False):
                    context_returns = {
                        "routing_infos": context.routing_infos,
                        "routing_gates": context.routing_gates,
                    }
                    return results, context_returns
            return results

        return wrapper_func
