import re


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
    from mttl.models.containers.base import ExpertContainer

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
    Combine modify_modules and modify_layers into a single regex
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
