import sys
import os
import tqdm
import torch
import hashlib
import nevergrad as ng
from string import Template
from torch.utils.data import DataLoader
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.dataloader.ni_metrics import compute_metrics
from mttl.evaluators.base import compute_task_aggregation

from src.config import ExpertConfig
from huggingface_hub import login
from src.expert_model import MultiExpertModel
from mttl.datamodule.collators import DefaultDataModule
from mttl.utils import logger, setup_logging
from mttl.vllm_engines.engines import LLMEngineMMLU, free_memory


def mmlu_get_loss(model, dm: DefaultDataModule, graph_string, use_vllm=True):
    # use gpu if available
    train_loss = 0
    if use_vllm:
        # use vllm
        model_hash = hashlib.sha256()
        model_hash.update(f"{graph_string}_{model.model.__class__}".encode())
        model = LLMEngineMMLU(
            model,
            temp_path=f"{os.environ.get('MTTL_TEMP','/tmp/merged')}/{model_hash.hexdigest()}/",
        )
        dataloader: DataLoader = dm.val_dataloader()
        all_predictions, all_references, all_task_names = model.eval(
            dataloader,
            top_p=0.9,
            temperature=0.9,
            max_tokens=100,
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


def get_score(
    weights, dm, template: Template, basemodel: MultiExpertModel, get_loss, get_regular
):
    graph_string = template.safe_substitute(weght1=weights[0], weght2=weights[1])
    logger.info(f"Loading {graph_string} into the model")
    basemodel.load_from_graph_string(graph_string, action="merge")
    # minimize the metric
    loss = get_loss(basemodel, dm, graph_string)
    # L1 regularization term
    metric_val = loss + get_regular(weights)
    return metric_val


def main(s: Template, dm: DefaultDataModule, model):
    _get_score = partial(
        get_score,
        dm=dm,
        template=s,
        get_loss=mmlu_get_loss,
        basemodel=model,
        get_regular=default_l1_regularization,
    )

    instrum = ng.p.Array(
        init=[0] * 2,
        upper=[1.5] * 2,
        lower=[-1.5] * 2,
    )

    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=2)
    recommendation = optimizer.minimize(_get_score)
    print(recommendation.value)


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

    s = Template(
        "security_studies -> linear(sordonia/expert_llama2_13b_security_studies:$weght1);\
        abstract_algebra -> linear(sordonia/expert_llama2_13b_security_studies:$weght2);"
    )

    dm = MMLUDataModule(config, for_generation=True, do_tokenize=False)

    model_class = MultiExpertModel
    module = model_class(**vars(config), tokenizer=dm.tokenizer)

    main(s, dm=dm, model=module)
