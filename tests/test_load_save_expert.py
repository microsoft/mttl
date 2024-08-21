from mttl.models.modifiers.lora import LoRAConfig


def test_load_expert_from_checkpoint(tmp_path):
    from mttl.models.expert_model import ExpertModel, ExpertModelConfig
    from mttl.models.library.expert_library import LocalExpertLibrary

    model = ExpertModel(
        ExpertModelConfig(
            "EleutherAI/gpt-neo-125m",
            expert_name="a",
            modifier_config=LoRAConfig(modify_layers="k_proj"),
        )
    )
    model.save_pretrained(tmp_path)

    library = LocalExpertLibrary(tmp_path)
    library.add_expert_from_ckpt(tmp_path)

    assert len(library) == 1
    assert library["a"] is not None
