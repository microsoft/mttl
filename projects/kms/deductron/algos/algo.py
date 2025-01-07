from dataclasses import dataclass
from typing import Dict, List

import numpy as np


class Algo:
    @classmethod
    def add_parser_args(self, parser):
        return parser


@dataclass
class Request:
    """Generic box to store info about an example."""

    # id of the prompt, or query
    query_id: int = None
    # messages to be sent to the model
    messages: List[Dict[str, str]] = None
    # response from the model
    response: str = None
    # label for the response
    label: str = None
    # whether the response was finished
    finished: bool = False
    # reward for the response
    reward: float = None


class RequestUtils:
    @classmethod
    def populate(cls, requests: List[Request], values: List, key: str):
        if len(requests) != len(values):
            raise ValueError("requests and values must have the same length")

        if not hasattr(requests[0], key):
            raise ValueError(f"requests must have attribute {key}.")

        for request, value in zip(requests, values):
            setattr(request, key, value)

    @classmethod
    def gather_max_avg_reward(cls, requests: List[Request]):
        rewards_by_query_id = cls.group_by_query_id(requests, "reward")
        all_rewards = [r.reward for r in requests]
        max_rewards = [max(rewards) for rewards in rewards_by_query_id.values()]
        return np.mean(max_rewards), np.mean(all_rewards)

    @classmethod
    def group_by_query_id(cls, requests: List[Request], key: str = None):
        if key and not hasattr(requests[0], key):
            raise ValueError(f"requests must have attribute {key}.")

        grouped = {}
        for request in requests:
            if key:
                value = getattr(request, key)
            else:
                value = request

            if request.query_id not in grouped:
                grouped[request.query_id] = []

            grouped[request.query_id].append(value)
        return grouped

    @classmethod
    def group_by_field(cls, requests: List[Request], field: str):
        if not hasattr(requests[0], field):
            raise ValueError(f"requests must have attribute {field}.")

        grouped = {}
        for request in requests:
            value = getattr(request, field)
            if value is None:
                raise ValueError(f"{field} cannot be None.")
            if value not in grouped:
                grouped[value] = []
            grouped[value].append(request)
        return grouped
