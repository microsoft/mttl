import re
import torch
from typing import Dict
import sys
import os
import re
from string import Template
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from mttl.models.utils import download_from_hub
from mttl.utils import get_checkpoint_path, logger
from projects.wiki_experts.src.config import ExpertConfig
from dataclasses import dataclass
from typing import Union


@dataclass
class Expert:
    expert_config: ExpertConfig
    expert_weights: Dict[str, torch.Tensor]


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self._cached_instantiation = None

    @classmethod
    def from_args(cls, name, graph, args=None):
        return Node(name)

    def instantiate(self, *args, **kwargs):
        if self._cached_instantiation is not None:
            return self._cached_instantiation

        assert (
            len(self.children) <= 1
        ), "Node can only have one child for now, use operators instead."

        instantiation = []
        if not self.children:
            # consider this to be a leaf node
            instantiation = [load_expert(self.name)]
        else:
            instantiation = [self.children[0].instantiate(*args, **kwargs)[0]]
        return instantiation

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class OperatorNode(Node):
    def __init__(self, name):
        super().__init__(name)

    @classmethod
    def from_args(cls, args, graph):
        raise NotImplementedError


class LinearNode(OperatorNode):
    @classmethod
    def from_args(cls, name, graph, args=None):
        node = LinearNode(name)
        node.weights = []

        node_args_pairs = args.split(",")
        for pair in node_args_pairs:
            child_name, weight = pair.split(":")
            node.children.append(graph.get_or_create_node(child_name.strip()))
            node.weights.append(float(weight.strip()))
        return node

    def instantiate(self, *args, **kwargs):
        if self._cached_instantiation is not None:
            return self._cached_instantiation

        instantiation = []

        for node in self.children:
            instantiation.append(node.instantiate(*args, **kwargs)[0])

        # now, merge with a given importance weight
        assert len(instantiation) == len(self.weights)

        merged_weights = {}
        for expert, weight in zip(instantiation, self.weights):
            for k, v in expert.expert_weights.items():
                value = v * torch.tensor(weight, dtype=v.dtype)
                if k in merged_weights:
                    merged_weights[k] += value
                else:
                    merged_weights[k] = value

        return [
            Expert(
                expert_config=instantiation[0].expert_config,
                expert_weights=merged_weights,
            )
        ]

    def __repr__(self):
        return "linear({})".format(
            ", ".join(["{}:{}".format(n, w) for n, w in zip(self.nodes, self.weights)])
        )


class GraphTemplate:
    """
    The purpose of the class is to dynamically generate module graphs with different weights
    """

    def __init__(self, template: Union[Template, str]):
        template = template.template if isinstance(template, Template) else template
        modules = template.split(";")
        self.name_to_modulestring = {}
        self.name_to_parameters = defaultdict(list)
        self._parameters = []
        template = ""
        for t in modules:
            if len(t.strip()) > 0:
                module_name = t.split("->")[0].strip()
                module_string = t.strip()
                variables = re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", module_string)

                # extract parameters from module_string: everythin that starts with $
                for i, v in enumerate(variables):
                    # change variable to follow pattern $weight_{module_name}_{i}
                    new_v = f"weight_{module_name}_{i}"
                    self.name_to_parameters[module_name].append(new_v)
                    self._parameters.append(new_v)
                    module_string = module_string.replace(v, new_v)
                template += f"{module_string};"
                self.name_to_modulestring[module_name] = module_string
        self.template = Template(template)

    @property
    def parameters(self):
        return self._parameters

    def to_graph_string(self, d: dict):
        return self.template.substitute(d)

    def to_graph(self, d: dict):
        template = self.to_graph_string(d)
        return ModuleGraph.from_string(template)

    def __len__(self):
        return len(self.name_to_modulestring.keys())


class ModuleGraph:
    # Operator-to-class mapping
    OPERATOR_CLASSES = {None: Node, "linear": LinearNode}

    def __init__(self):
        self.nodes = {}

    def get_or_create_node(self, node_name, node_type=None, args=None):
        if node_name not in self.nodes:
            node_class = self.OPERATOR_CLASSES[node_type]
            self.nodes[node_name] = node_class.from_args(node_name, self, args)
        return self.nodes[node_name]

    def dumps(self):
        graph_str = []
        for node_name, node in self.nodes.items():
            if not node.children:
                continue
            if isinstance(node, OperatorNode):
                continue
            graph_str.append(
                "{} -> {}".format(node_name, ", ".join([n.name for n in node.children]))
            )
        return "; ".join(graph_str)

    @classmethod
    def from_string(self, s):
        graph = ModuleGraph()
        parts = [p.strip() for p in s.split(";")]

        for part in parts:
            if "->" in part:
                source, targets = part.split("->")
                targets = targets.strip()
                source = source.strip()

                match_source = re.match(r"(\w+)\((.+)\)", source.strip())
                if match_source:
                    raise ValueError("Source cannot be an operator.")

                match_target = re.match(r"(\w+)\((.+)\)", targets.strip())
                source_node = graph.get_or_create_node(source)

                if match_target:  # This means there's an operator
                    operator = match_target.group(1)
                    args = match_target.group(2)

                    if operator not in self.OPERATOR_CLASSES:
                        raise ValueError(
                            f"Unknown operator: '{operator}' in segment '{part}'"
                        )

                    children = [
                        graph.get_or_create_node(
                            node_name=targets, node_type=operator, args=args
                        )
                    ]
                else:
                    children = [
                        graph.get_or_create_node(t.strip()) for t in targets.split(",")
                    ]
                source_node.children.extend(children)
        return graph

    @property
    def roots(self):
        parent_nodes = {}
        for _, parent_node in self.nodes.items():
            for children in parent_node.children:
                parent_nodes[children] = parent_node
        return set(self.nodes.values()) - set(parent_nodes.keys())

    @property
    def leaves(self):
        children_nodes = set()
        for _, parent_node in self.nodes.items():
            if not parent_node.children:
                children_nodes.add(parent_node)
        return children_nodes

    def create_modules(self, *args, **kwargs):
        root_modules = {}
        for root in self.roots:
            root_modules[root.name] = root.instantiate(*args, **kwargs)[0]
        return root_modules


def load_expert(
    expert_path: str,
    expert_name: str = None,
):
    # load the expert weights
    import os

    if os.path.isfile(expert_path):
        expert_checkpoint = get_checkpoint_path(expert_path)
    else:
        expert_checkpoint = download_from_hub(expert_path)

    logger.info(f"Loading expert from {expert_checkpoint}...")
    expert_checkpoint = torch.load(expert_checkpoint, map_location="cpu")

    expert_config = ExpertConfig(
        kwargs=expert_checkpoint["hyper_parameters"], silent=True, raise_error=False
    )

    expert_name = expert_name or expert_config.expert_name
    if expert_name is None:
        if expert_config.finetune_task_name is not None:
            expert_name = expert_config.finetune_task_name
        else:
            expert_name = os.path.basename(expert_path)
        logger.info(
            "Assigning expert name, not found in checkpoint: {}".format(expert_name)
        )

    expert_config.expert_name = expert_name

    expert_weights = expert_checkpoint["state_dict"]
    expert_weights = {k.replace("model.", "", 1): v for k, v in expert_weights.items()}
    return Expert(expert_config, expert_weights)


if __name__ == "__main__":
    # Example usage:
    s = """
    security_studies -> B;
    B -> linear(sordonia/llama2-13b-platypus:0.5, sordonia/expert_llama2_13b_security_studies:3);
    C -> linear(B:0.5);
    default -> C
    """
    s = """
    security_studies -> linear(sordonia/expert_llama2_13b_security_studies:1);
    """

    graph = ModuleGraph.from_string(s)
    print(graph)
    print(graph.roots)
    print(graph.leaves)
    print(graph.dumps())
    print(graph.create_modules().keys())
