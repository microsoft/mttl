import os
import re
import sys
from huggingface_hub import login
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.evaluators import MMLUEvaluator
from mttl.utils import setup_logging, logger

# register models
from projects.wiki_experts.src.expert_model import MultiExpertModel
from projects.wiki_experts.src.config import ExpertConfig
from mttl.vllm_engines.engines import free_memory


base_dir = os.environ.get(
    "MODULES_DIR", "/home/v-oostapenko/dev/amlt/wiki_experts_train_per_cat_2_if/"
)
base_dir_tempalte = (
    lambda subject: f"{base_dir}/ll2_13b_expert_2_{subject}__qa-ostapeno_qa-openai_icl5_clen128_maxD-1_maxC8000_0_length_matched___5e-5_/{subject}/meta-llama_Llama-2-13b-hf-mmlu_test_oracle"
)


def run_eval(args):
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # module_graph, tasks_to_module = get_module_gaph(args.module_graph)
    # print(module_graph)

    # We can do:
    #   - in-distribution evaluation: test sets we consider are the test sets of the tasks we have experts for
    #   - out-of-distribution evaluation: new task

    # Questions:
    # 1. How good is the merging optimization procedure?
    # On a the in-domain val-set of one of the modules in the population, can it converge to the right routing? (run this for each of the 10 test sets)
    # Does it attain perofrmance like the in-domain module? Could it find this module? if not, did it find a better combination?
    # How does it compare to join-training?

    # 2. How well can we generalize to new task? the baseline here is using jointly pre-trained model vs. merging the experts
    # If I could now tain on the new task a bit, is it bette to use as innitialization the merged pexpert vs. jointl pre-trained?

    # Given the modules lets first eval all of them on each other's test sets -> get a tansfe matix

    from projects.wiki_experts.src.graph.module_graph import ModuleGraph

    # instantiate weights
    weight_names = re.compile(r"\$weight_[a-zA-Z0-9_]+").findall(args.module_graph)
    for weight_name in weight_names:
        args.module_graph = args.module_graph.replace(weight_name, "1.0")

    print(args.module_graph)

    graph = ModuleGraph.from_string(args.module_graph)
    tasks = [node.name for node in graph.roots]

    args.finetune_task_name = ",".join(tasks)
    mmlu = MMLUEvaluator(args, split=args.mmlu_test_split, use_vllm=True)
    module = MultiExpertModel(**vars(args), tokenizer=mmlu.datamodule.tokenizer)
    module.load_from_graph(graph, action="merge")
    scores = mmlu.evaluate(module)

    del module, mmlu
    free_memory()
    print(scores)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
