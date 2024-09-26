import re
from typing import Tuple

from mttl.logging import logger, warn_once
from mttl.models.containers.base import ExpertContainer
from mttl.models.containers.lora_containers import (
    CoalescedLoRAExpertContainer,
    LoRAExpertContainer,
)
from mttl.models.containers.peer_container import PEERMLPContainer
from mttl.models.containers.selectors.base import (
    Selector,
    SelectorConfig,
    SelectorsCache,
    SelectorView,
    get_selector,
)
from mttl.models.library.expert import Expert
from mttl.models.modifiers.base import Modifier
from mttl.utils import logger


def _extract_identifier(string, match_on="finegrained"):
    """Returns a unique identifier for the "chunk" of layers sharing the
    same underlying selector

    e.g. 'block' : 'encoder.block.0.layer.0.SelfAttention' -> 'encoder.block.0'
    """
    if match_on == "finegrained" or match_on == "*":
        return string
    if match_on == "coarsegrained":
        return "shared"
    pos = string.find(f"{match_on}")
    if pos == -1:
        raise ValueError(
            "Cannot resolve the selected router granularity: %s" % match_on
        )
    return string[: pos + len(match_on)]


def get_container_class(modifier: str):
    import os

    if modifier == "lora":
        if os.environ.get("COALESCED_LORA_CONTAINER", "False") == "1":
            return CoalescedLoRAExpertContainer
        return LoRAExpertContainer
    elif modifier == "skilled_lora":
        if not os.environ.get("COALESCED_LORA_CONTAINER", "False") == "1":
            warn_once(
                "COALESCED_LORA_CONTAINER is not set to 1, but still using it for SkilledLoRA"
            )
        return CoalescedLoRAExpertContainer
    elif modifier == "peer":
        return PEERMLPContainer
    else:
        raise ValueError(f"Cannot find modifier: {modifier}")


def filter_expert_weights(layer_name, expert_weights):
    # subset the relevant expert weights starting w __layer_name__
    keys = list(expert_weights.keys())

    if "transformer.h" in keys[0] and "layers." in layer_name:
        # phi-huggingface to phi-private
        weights = {}
        for k, v in expert_weights.items():
            k = k.replace("transformer.h.", "layers.")
            ks = k.split(".")
            ks[1] = int(ks[1]) + 1
            k = ".".join(map(str, ks))
            weights[k] = v
    else:
        weights = expert_weights

    weights = {
        k.replace(layer_name + ".", ""): v
        for k, v in weights.items()
        if k.startswith(layer_name)
    }
    if not weights:
        return None
    return weights


def create_selector_for_container(
    transformer,
    container,
    modifier_name: str,
    selector_config: SelectorConfig = None,
    selector_cache: SelectorsCache = None,
) -> Selector:
    if container.selector is not None and container.selector.config == selector_config:
        # selector already exists and has the same config
        return

    identifier = _extract_identifier(
        container.layer_name, selector_config.router_granularity
    )

    # we create a new selector if it doesn't exist for this identifier, or
    # if we are replacing a previous one of a different type
    create_new_selector = (
        not selector_cache.get(modifier_name, identifier)
        or selector_cache.get(modifier_name, identifier).config != selector_config
    )
    if create_new_selector:
        # Special case when you have a decoder layer in an enc-dec model
        selector = get_selector(
            selector_config,
            layer=container.layer,
        )
        selector_cache.insert(modifier_name, identifier, selector)

        # selector needs to know how many times it will be called per forward pass in order to be able to reset the cache
        selector.total_calls_per_forward += 1
    else:
        selector: Selector = selector_cache.get(modifier_name, identifier)
        # selector needs to know how many times it will be called per forward pass in order to be able to reset the cache
        selector.total_calls_per_forward += 1
        selector = selector.create_view()

    # assign this selector to the container, propagates expert informations, etc.
    container.assign_selector(selector)
    return selector


def replace_selector_for_container(
    transformer,
    modifier_name: str,
    selector_config: SelectorConfig = None,
    selector_cache: SelectorsCache = None,
    force_replace: bool = False,
) -> Tuple[int, int]:
    """
    Assigns a selector to the expert containers in the transformer model.
    """
    expert_containers = []
    for _, module in dict(transformer.named_modules()).items():
        if isinstance(module, ExpertContainer):
            # check if the container holds the same modifier type, e.g. PEERConfig --> "peers"
            for supports_config in module.__supports_configs__:
                container_modifier = Modifier.get_name_by_config_class(supports_config)
                # selector does not apply to this container
                if not container_modifier == modifier_name:
                    continue
                else:
                    expert_containers.append(module)

        for _, layer in dict(module.named_children()).items():
            if isinstance(layer, ExpertContainer):
                # check if the container holds the same modifier type, e.g. LoRAConfig --> "lora"
                for supports_config in layer.__supports_configs__:
                    container_modifier = Modifier.get_name_by_config_class(
                        supports_config
                    )
                    # selector does not apply to this container
                    if not container_modifier == modifier_name:
                        continue
                    else:
                        expert_containers.append(layer)
                        break

    if not expert_containers:
        raise ValueError(
            f"No expert containers found for modifier type: {modifier_name}. Cannot assign a selector! Load some experts beforehand."
        )

    if force_replace:
        for container in expert_containers:
            container.selector = None
        selector_cache.clear(modifier_name)

    n_selectors = 0
    n_selectors_views = 0

    for container in expert_containers:
        selector = create_selector_for_container(
            transformer,
            container,
            modifier_name,
            selector_config,
            selector_cache,
        )
        n_selectors += isinstance(selector, Selector)
        n_selectors_views += isinstance(selector, SelectorView)

    return n_selectors, n_selectors_views


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, sequence):
        word = sequence.split(".")
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, sequence):
        word = sequence.split(".")
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, sequence):
        node = self.root
        prefix = sequence.split(".")
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def has_leaf_prefix(self, sequence):
        node = self.root
        prefix = sequence.split(".")
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
            if node.is_end_of_word:
                return True
        return False

    def print_all_words(self):
        self._print_all_words_helper(self.root, "")

    def _print_all_words_helper(self, node, prefix):
        if node.is_end_of_word:
            print(prefix)

        for char, child_node in node.children.items():
            self._print_all_words_helper(child_node, prefix + char)


def get_modifiable_modules(transformer):
    """Get modules to modify in the transformer model.
    Filter out modules that are inside expert containers."""
    trie = Trie()
    for m_name, module in dict(transformer.named_modules()).items():
        # if m_name is ExpertContainer, insert to the trie
        if isinstance(module, ExpertContainer):
            trie.insert(m_name)
    for m_name, module in dict(transformer.named_modules()).items():
        # for all the sub modules in the trie, skip if it is inside an expert container
        if not trie.search(m_name) and trie.has_leaf_prefix(
            m_name
        ):  # it indicate the m_name is from the expert container
            continue
        yield m_name, module


def create_modif_regex(modify_modules, modify_layers=None):
    """
    Combine modify_modules and modify_layers into a single regex to keep add_expert_to_transformer slim
    """
    is_set = lambda x: x is not None and x != ""

    if not is_set(modify_modules) and not is_set(modify_layers):
        raise ValueError(
            "Neither modify_modules nor modify_layers are set, will not modify anything"
        )

    if is_set(modify_modules) and not is_set(modify_layers):
        return modify_modules
    if not is_set(modify_modules) and is_set(modify_layers):
        return modify_layers

    # keep backward compatibility
    modules = modify_modules.split("|")
    layers = modify_layers.split("|")
    parts = []
    for m in modules:
        for l in layers:
            if m == ".*":
                l.replace(".*", "")
            parts.append(f"{m}\\.{l}")
    return "|".join(parts)


def match_modules_to_modify(transformer, modify_modules):
    """
    Match modules in the transformer model based on the modify_modules regex
    """
    for m_name, module in get_modifiable_modules(transformer):
        if re.fullmatch(modify_modules, m_name):
            yield m_name, module


def add_expert_to_transformer(
    transformer,
    expert: Expert,
    action: str = "route",
    is_default: bool = False,
    selector_config: SelectorConfig = None,
    selector_cache: SelectorsCache = None,
) -> None:
    """
    Routine to add an expert to the transformer architecture.

    Params:
        transformer: the transformer model to modify
        expert: expert instance that needs to be added
        action: whether to route or merge this expert, default is `route`
        is_default: whether the expert should be set as default
        selector_config: selector configuration to use for the model
        selector_cache: cache to store the selectors for the model
    """
    expert_config = expert.expert_config

    if not expert.name:
        raise ValueError("Expert name cannot be empty!")

    from mttl.models.containers.hard_prompts_container import (
        add_hard_prompt_to_transformer,
    )
    from mttl.models.modifiers.modify_model import get_modifier_name

    model_modifier = get_modifier_name(expert_config)

    if model_modifier == "hard_prompt":
        return add_hard_prompt_to_transformer(
            transformer,
            expert,
            action=action,
            is_default=is_default,
        )

    total_layers = 0
    added_layers = []
    added_containers = []

    modify_modules = create_modif_regex(
        expert_config.modify_modules, expert_config.modify_layers
    )
    for m_name, module in match_modules_to_modify(transformer, modify_modules):
        # no layers to modify, try modifying the module
        total_layers += 1
        module_name = f"{m_name}"

        if not isinstance(module, ExpertContainer):
            CONTAINER_CLASS = get_container_class(model_modifier)
            expert_container = CONTAINER_CLASS(
                expert_config,
                module,
            )
            expert_container.__layer_name__ = module_name

            parent_name, child_name = m_name.rsplit(".", 1)
            parent_module = dict(transformer.named_modules())[parent_name]
            setattr(
                parent_module,
                child_name,
                expert_container,
            )
            added_containers.append(expert_container)
        else:
            expert_container = module

        added_layers.append(expert_container.__layer_name__)
        expert_container.add_expert(
            expert,
            action=action,
            is_default=is_default,
        )
    ### PARAM TYING ###
    # Note: because experts are added into expert containers
    # instead of parameter names being e.g. model.layers.4.self_attn.q_proj.lora_a,
    # it will be model.layers.4.self_attn.q_proj.experts.module1.lora_a

    # For this reason tying with q_proj\\.lora_a|k_proj\\.lora_a|v_proj\\.lora_a will not work,
    # and it has to be q_proj.*\\.lora_a|k_proj.*\\.lora_a|v_proj.*\\.lora_a
    from mttl.models.modifiers.base import get_target_2_source_param_mapping, tie_params

    target_2_source_param = get_target_2_source_param_mapping(
        transformer.named_parameters(), expert_config.tie_params
    )
    tie_params(transformer, expert_config, target_2_source_param)

    if not added_layers:
        raise ValueError(
            "You were trying to add an expert but no expert containers were created, this is likely due to a misconfiguration of the expert config."
            " `modify_layers` and `modify_modules` did not return a match for the current model."
        )

    if selector_config is not None:
        replace_selector_for_container(
            transformer,
            model_modifier,
            selector_config,
            selector_cache,
        )

        if not selector_cache.get(model_modifier):
            raise ValueError(
                "No selectors were created but a routing config was specified. Check your selector_config and model architecture."
            )

        logger.debug(
            "Added expert %s, with %s selectors",
            expert.name,
            len(selector_cache.get(model_modifier)),
        )

    logger.debug("Patched layers: %s", added_layers)
