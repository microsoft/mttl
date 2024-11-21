from mttl.models.library.expert_library import ExpertLibrary
from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.models.containers.selectors import TaskNameSelectorConfig, ArrowSelectorConfig
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
import torch
import os

# library = ExpertLibrary.get_expert_library("hf://zhan1993/gpt_debug-epoch_0")
# an_expert = library[next(iter(library.keys()))]
# config = ExpertConfig.from_dict(an_expert.training_config)
# breakpoint()


if not os.path.exists("trained_gpt125m_experts_colab"):
    library = ExpertLibrary.get_expert_library(
        "hf://sordonia/trained_gpt125m_experts_colab"
    )

    # let's create a local clone of this library! This step is not necessary if we trained experts above!
    library = library.clone("local://trained_gpt125m_experts_colab")
else:
    library = ExpertLibrary.get_expert_library("local://trained_gpt125m_experts_colab")

# test the in-distribution results
config = ExpertConfig.from_json(
    "projects/modular_llm/configs/models/gptneo_125m_fast.json"
)
config.finetune_task_name = ",".join(
    [
        "wiqa_what_is_the_final_step_of_the_following_process,sciq_Multiple_Choice",
        "adversarial_qa_droberta_answer_the_following_q",
        "duorc_SelfRC_question_answering",
        "cos_e_v1_11_description_question_option_id",
        "wiki_qa_Is_This_True_",
        "quail_description_context_question_text",
        "wiki_hop_original_explain_relation",
    ]
)
datamodule = get_datamodule(config, for_generation=True)
evaluator = RougeEvaluator(datamodule)

device_map = "cuda" if torch.cuda.is_available() else "cpu"
print("Test examples:", len(datamodule.test_dataset))

# the oracle model uses the "task name" to generate!
# model = MultiExpertModel.from_pretrained_library(
#     "local://trained_gpt125m_experts_colab",
#     selector_config=TaskNameSelectorConfig(),
#     device_map=device_map,
# )

# oracle_rouge = evaluator.evaluate(model, split="test")


# model.set_selector("lora", UniformSelectorConfig())
# uniform_mbc_rouge = evaluator.evaluate(model, split="test")

from mttl.evaluators.base import EvaluatorRunner, setup_evaluators

evaluation_task = "arc-easy"

config.model = "google/flan-t5-large"
config.model_family = "seq2seq"
with torch.no_grad():
    runner: EvaluatorRunner = setup_evaluators(
        model_type=config.model,
        model_family=config.model_family,
        max_input_length=config.max_input_length,
        max_output_length=config.max_output_length,
        predict_batch_size=config.predict_batch_size,
        truncation_side=config.truncation_side,
        tasks=evaluation_task,
    )

# test the out-of-distribution results

# model = MultiExpertModel(
#     MultiExpertModelConfig(base_model="google/flan-t5-large"),
#     device_map="cuda",
#     precision="fp16",
# )
# base_perf = runner.run(model)

model = MultiExpertModel.from_pretrained(
    "./trained_flan_t5_large_arrow_model", device_map="cuda", precision="fp16"
)
arrow_perf = runner.run(model)
# breakpoint()
