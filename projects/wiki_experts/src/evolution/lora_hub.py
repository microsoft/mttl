import sys
import os
import tqdm
import copy
import torch
import hashlib
import nevergrad as ng

from torch.utils.data import DataLoader
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from typing import Callable, Dict
from mttl.dataloader.ni_metrics import compute_metrics
from mttl.evaluators.base import compute_task_aggregation

from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.expert_model import MultiExpertModel
from mttl.utils import logger, setup_logging
from projects.wiki_experts.src.graph.module_graph import ModuleGraph
from mttl.vllm_engines.engines import LLMEngineMMLU, free_memory
from mttl.evaluators import MMLUEvaluator


def mmlu_get_loss(
    model: MultiExpertModel,
    tokenizer,
    dataloader: DataLoader,
    graph_string,
    use_vllm=True,
    use_loss=False,
):
    # use gpu if available
    train_loss = 0
    if use_vllm:
        # use vllm
        generation_config = model.generation_config
        model_hash = hashlib.sha256()
        model_hash.update(f"{graph_string}_{model.model.__class__}".encode())
        model = LLMEngineMMLU(
            model,
            temp_path=f"{os.environ.get('MTTL_TEMP','/tmp/merged')}/{model_hash.hexdigest()}/",
        )
        all_predictions, all_references, all_task_names = model.eval(
            dataloader,
            generation_config=generation_config,
            tokenizer=tokenizer,
        )
        del model
        free_memory()

        eval_metrics = compute_metrics(
            all_predictions, [[r] for r in all_references], reduction="none"
        )
        return (
            compute_task_aggregation(all_task_names, eval_metrics["exact_match"])[
                "all"
            ]["mean"]
            * -1.0
        )

    else:
        if use_loss:
            with torch.no_grad():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    with torch.no_grad():
                        loss = model(batch)
                    train_loss += loss.detach().float()
            loss = train_loss.float()
            # average loss over the number of examples
            return float(loss) / len(dataloader.dataset)
        else:
            # using accuracy
            mmlu_evaluator = MMLUEvaluator(model.hparams, split="test", use_vllm=False)
            scores = mmlu_evaluator.evaluate(model, dataloader=dataloader)
            return scores["all"]["mean"] * -1.0


def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


class RoutingOptimizer:
    def __init__(
        self,
        model: MultiExpertModel,
        modules_2_dest: Dict,
        dataloader: DataLoader,
        get_loss: Callable,
        budget=5,
        task_name="new_task",
    ) -> None:
        self.task_name = task_name
        self.model: MultiExpertModel = model
        self.module_graph: ModuleGraph = self.construct_graph(modules_2_dest)
        self.varaibles = self.module_graph.get_variables()
        self.K = len(self.varaibles)

        self.parametrization = ng.p.Array(
            init=[0] * self.K,
            upper=[1.5] * self.K,
            lower=[-1.5] * self.K,
        )
        self.dataloader = dataloader
        self.optimizer = ng.optimizers.NGOpt(
            parametrization=self.parametrization, budget=budget
        )
        self.get_loss = get_loss

    def construct_graph(self, modules_to_dest: Dict):
        s = f"{self.task_name} -> linear("
        for i, (name, destination) in enumerate(modules_to_dest.items()):
            s += f"{destination}:${i},"
        s = s[:-1] + ")"
        graph = ModuleGraph.from_string(s)
        return graph

    def get_graph_vars(self, weights: list):
        s = {}
        for p, w in zip(self.varaibles, list(weights)):
            s[p] = w
        return s

    def optimize(
        self,
    ):
        def get_score(
            weights,
            dataloader,
            basemodel: MultiExpertModel,
            get_loss,
            get_regular,
            get_vars,
        ):
            graph_vars = get_vars(weights)
            graph_string = self.module_graph.dumps(**graph_vars)
            logger.info(f"Testing weights {graph_string} into the model")
            model = copy.deepcopy(basemodel)
            model.load_from_graph(self.module_graph, action="merge", **graph_vars)
            # minimize the metric
            loss = get_loss(
                model=model,
                tokenizer=model.tokenizer,
                dataloader=dataloader,
                graph_string=graph_string,
            )
            # L1 regularization term
            metric_val = loss + get_regular(weights)
            return metric_val

        _get_score = partial(
            get_score,
            dataloader=self.dataloader,
            get_loss=self.get_loss,
            basemodel=self.model,
            get_regular=default_l1_regularization,
            get_vars=self.get_graph_vars,
        )
        recommendation = self.optimizer.minimize(_get_score)
        logger.info(recommendation.value)

        best_vars = self.get_graph_vars(recommendation.value)
        best_graph = self.module_graph.dumps(**best_vars)
        return recommendation.value, best_graph


if __name__ == "__main__":
    from mttl.datamodule.mmlu_data_module import MMLUDataModule

    setup_logging()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = ExpertConfig()
    config.model = "meta-llama/Llama-2-13b-hf"
    config.load_in_8bit = False
    config.model_family = "gpt"
    config.data_dir = os.environ["MMLU_DATA_DIR"]
    config.predict_batch_size = 2
    config.max_input_length = 4096
    config.max_output_length = 5
    # config.model_modifier = "lora"
    config.modify_modules = ".*mlp.*"
    config.modify_layers = "gate_proj|down_proj|up_proj"
    config.predict_batch_size = 1
    config.finetune_task_name = "abstract_algebra"

    modules_2_dest = {
        "security_studies": "sordonia/expert_llama2_13b_security_studies",
        "platy": "sordonia/llama2-13b-platypus",
    }
    use_vllm = False
    dm = MMLUDataModule(config, for_generation=use_vllm)
    module = MultiExpertModel(
        **vars(config), tokenizer=dm.tokenizer, device_map="cpu" if use_vllm else "auto"
    )

    _mmlu_get_loss = partial(mmlu_get_loss, use_vllm=use_vllm)

    optimizer = RoutingOptimizer(
        model=module,
        modules_2_dest=modules_2_dest,
        dataloader=dm.test_dataloader(),
        get_loss=_mmlu_get_loss,
        budget=2,
        task_name=config.finetune_task_name,
    )
    recommendation = optimizer.optimize()
    print(recommendation)
