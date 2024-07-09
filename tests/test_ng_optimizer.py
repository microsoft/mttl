import numpy as np
import pytest
from projects.modular_llm.src.nevergrad_opt import NGRoutingOptimizer
from mttl.models.expert_model import ExpertModel as ExpertTrainer, MultiExpertModel
from mttl.models.expert_config import ExpertConfig
from mttl.models.library.expert_library import LocalExpertLibrary


# remove this for now, since NG Routing is be to rebuilt.
def test_NGRoutingOptimizer(tmp_path, make_tiny_llama, create_dummy_expert):
    config = ExpertConfig(
        kwargs={
            "model_modifier": "lora",
            "modify_layers": "gate_proj|down_proj|up_proj",
            "modify_modules": ".*mlp.*",
            "trainable_param_names": ".*lora_[ab].*",
            "output_dir": tmp_path,
            "model": "",
        }
    )
    # create random Lora
    expert1 = create_dummy_expert(config, "module1")
    expert2 = create_dummy_expert(config, "module2")

    get_loss = lambda *args, **kwargs: 0.0

    library = LocalExpertLibrary(tmp_path)
    library.add_expert(expert1, expert1.name)
    library.add_expert(expert2, expert2.name)
    model_object = make_tiny_llama()
    model = MultiExpertModel(model_object=model_object, **vars(config))

    # create an NGRoutingOptimizer instance
    optimizer = NGRoutingOptimizer(
        model=model,
        expert_lib=library,
        get_loss=get_loss,
        budget=1,
        task_name="new_task",
        regularizer_factor=0.0,
    )

    # call the optimize method
    result = optimizer.optimize()

    # assert that the result is a tuple with two elements
    assert isinstance(result, tuple)
    assert len(result) == 2

    # assert that the first element of the result is a list
    print(result[0].__class__)
    assert isinstance(result[0], np.ndarray)

    # assert that the second element of the result is a string
    assert isinstance(result[1], dict)


if __name__ == "__main__":
    pytest.main([__file__])
