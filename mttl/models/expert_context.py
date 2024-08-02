import functools
import threading
from typing import List

from mttl.models.modifiers.routing import RoutingInfo


class InfoContainer:
    local = threading.local()
    local.context = None

    def __init__(self, model, routing_infos=None, **kwargs):
        self.model = model
        # stores the routing info for the model
        self._routing_infos = routing_infos
        # stores the routing gates for each layer, if any
        self._routing_gates = []

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
    def routing_infos(self, value: RoutingInfo):
        self._routing_infos = value

    @routing_gates.setter
    def routing_gates(self, value: List):
        self._routing_gates = value

    @classmethod
    def create(cls, model, routing_infos: RoutingInfo = None):
        """
        Creates a new context and sets it as the active context.
        """
        cls.local.context = cls(model, routing_infos)
        return cls.local.context

    @classmethod
    def wrap_forward(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a model class.
        """

        @functools.wraps(f)
        def wrapper_func(model, *args, **kwargs):
            if not isinstance(args[0], dict) and "input_ids" not in args[0]:
                raise ValueError(
                    "The first argument of the function to wrap must be a dictionary with 'input_ids' key."
                )

            InfoContainer.create(model, RoutingInfo.from_batch(args[0]))
            results = f(model, *args, **kwargs)
            return results

        return wrapper_func
