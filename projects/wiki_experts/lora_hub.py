import sys
import os
import tqdm
import copy
import torch
import hashlib
import nevergrad as ng

from torch.utils.data import DataLoader
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Union, Callable
from mttl.dataloader.ni_metrics import compute_metrics
from mttl.evaluators.base import compute_task_aggregation

from src.config import ExpertConfig
from huggingface_hub import login
from src.expert_model import MultiExpertModel
from mttl.datamodule.collators import DefaultDataModule
from mttl.utils import logger, setup_logging
from src.graph.module_graph import ModuleGraph
from mttl.vllm_engines.engines import LLMEngineMMLU, free_memory


def mmlu_get_loss(model, dm: DefaultDataModule, graph_string, use_vllm=True):
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
        dataloader: DataLoader = dm.val_dataloader()
        all_predictions, all_references, all_task_names = model.eval(
            dataloader,
            generation_config=generation_config,
            tokenizer=dm.tokenizer,
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
        dataloader: DataLoader = dm.val_dataloader()
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
        module_graph: ModuleGraph,
        dm: DefaultDataModule,
        get_loss: Callable,
        budget=5,
    ) -> None:
        self.model: MultiExpertModel = model
        self.module_graph = module_graph
        self.varaibles = module_graph.get_varaibles()
        self.K = len(self.varaibles)

        self.parametrization = ng.p.Array(
            init=[0] * self.K,
            upper=[1.5] * self.K,
            lower=[-1.5] * self.K,
        )
        self.dm = dm
        self.optimizer = ng.optimizers.NGOpt(
            parametrization=self.parametrization, budget=budget
        )
        self.get_loss = get_loss

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
            dm,
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
            loss = get_loss(basemodel, dm, graph_string)
            # L1 regularization term
            metric_val = loss + get_regular(weights)
            return metric_val

        _get_score = partial(
            get_score,
            dm=self.dm,
            get_loss=self.get_loss,
            basemodel=self.model,
            get_regular=default_l1_regularization,
            get_vars=self.get_graph_vars,
        )
        recommendation = self.optimizer.minimize(_get_score)
        logger.info(recommendation.value)
        return recommendation


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

    graph_string = "security_studies -> linear(sordonia/expert_llama2_13b_security_studies:$weight);\
                    abstract_algebra -> linear(sordonia/expert_llama2_13b_security_studies:$weight);\
                    abstract_algebra2 -> linear(sordonia/expert_llama2_13b_security_studies:2);"
    graph = ModuleGraph.from_string(graph_string)

    dm = MMLUDataModule(config, for_generation=True, do_tokenize=False)

    model_class = MultiExpertModel
    module = model_class(**vars(config), tokenizer=dm.tokenizer, device_map="cpu")

    optimizer = RoutingOptimizer(
        model=module,
        module_graph=graph,
        dm=dm,
        get_loss=mmlu_get_loss,
        budget=2,
    )
    recommendation = optimizer.optimize()
    print(recommendation)
