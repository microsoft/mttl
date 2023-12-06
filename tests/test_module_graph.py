from mttl.models.modifiers.expert_containers.module_graph import ModuleGraph


def test_module_graph():
    graph = ModuleGraph.from_string("a -> linear(b:1)")
    assert list(graph.roots)[0].name == "a"
