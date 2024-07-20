import functools
import threading

from mttl.models.modifiers.routing import RoutingInfo


class InfoContainer:
    local = None

    # thread-local storage that holds a stack of active contexts
    def __init__(self, model, routing_infos=None, **kwargs):
        self.model = model
        self._routing_infos = routing_infos

    @classmethod
    def get(cls):
        return cls.local

    @property
    def routing_infos(self):
        return self._routing_infos

    @routing_infos.setter
    def routing_infos(self, value: RoutingInfo):
        self._routing_infos = value

    @classmethod
    def create(cls, model, routing_infos: RoutingInfo = None):
        """
        Creates a new context and sets it as the active context.
        """
        cls.local = cls(model, routing_infos)
        return cls.local

    @classmethod
    def wrap(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a model class.
        """

        @functools.wraps(f)
        def wrapper_func(model, *args, **kwargs):
            InfoContainer.create(model, RoutingInfo.from_batch(args[0]))
            results = f(model, *args, **kwargs)
            return results

        return wrapper_func
