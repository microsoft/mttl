import sys
import os
import tqdm
import torch
import hashlib
import nevergrad as ng
from string import Template
from torch.utils.data import DataLoader
from functools import partial

sys.path.append("/home/v-oostapenko/dev/mttl")
sys.path.append("/home/v-oostapenko/dev/mttl/projects/wiki_experts/")
from mttl.datamodule.platypus_module import (
    PlatypusModule,
)
from src.graph.module_graph import ModuleGraph
from mttl.models.adapters import LoRA

from collections import OrderedDict
from src.config import ExpertConfig
from huggingface_hub import login
from src.expert_model import MultiExpertModel
from mttl.models.adapters import ExpertContainer
from vllm import LLM, SamplingParams
from mttl.datamodule.collators import DefaultDataModule, DatasetConfig
from src.data_transforms.engines import free_memory
from mttl.utils import logger, setup_logging


def save_merged_model(model, hf_path="/tmp/merged"):
    if os.path.exists(hf_path):
        return hf_path

    merged = []
    for name, module in model.model.named_modules():
        for c_name, child in module.named_children():
            if isinstance(child, LoRA) or isinstance(child, ExpertContainer):
                child.merge_with_layer()
                setattr(
                    module,
                    c_name,
                    child.layer,
                )
                merged.append(name)

    logger.info("Merged LoRA layers: %s" % merged)
    logger.info("Saving merged model to: %s" % hf_path)

    model.model.save_pretrained(hf_path, save_full_model=True)
    logger.info("Saving tokenizer to: %s" % hf_path)
    model.tokenizer.save_pretrained(hf_path)
    return hf_path


class _LLMEngineMMLU(LLM):
    def __init__(self, model, temp_path="/tmp/merged", **options):
        # merge adapters -- if needed --
        path = save_merged_model(model, hf_path=temp_path)
        options["model"] = path

        LLM.__init__(
            self, gpu_memory_utilization=0.8, disable_log_stats=False, **options
        )

    def get_losses(
        self,
        dataloader: DataLoader,
        top_p,
        temperature,
        max_tokens,
        tokenizer,
        **kwargs,
    ):
        labels = {}
        logprobs_for = 20
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=logprobs_for,
        )
        target_to_id = OrderedDict(
            {
                "A": tokenizer("A", add_special_tokens=False).input_ids[-1],
                "B": tokenizer("B", add_special_tokens=False).input_ids[-1],
                "C": tokenizer("C", add_special_tokens=False).input_ids[-1],
                "D": tokenizer("D", add_special_tokens=False).input_ids[-1],
            }
        )

        # we explicitly add requests here, so that we can keep track of the request id
        for request_id, batch in enumerate(
            tqdm.tqdm(dataloader, total=len(dataloader))
        ):
            for context, label in zip(batch["sources_texts"], batch["labels_texts"]):
                self.llm_engine.add_request(str(request_id), context, sampling_params)
                labels[str(request_id)] = label
        responses = self._run_engine(use_tqdm=True)
        responses_dict = {r.request_id: r for r in responses}
        # for each sample, for each token a list of logprobs of the logprobs_for most likely tokens
        gt = []
        pedictions = []
        pedictions_log_probs = []
        for r_id, response in responses_dict.items():
            gt.append(target_to_id[labels[r_id]])
            pedictions_log_probs.append(-torch.inf)
            pedictions.append(-100)
            logprobs_first_tok = response.outputs[0].logprobs[0]
            for _, tok_id in target_to_id.items():
                if (
                    tok_id in logprobs_first_tok
                    and logprobs_first_tok[tok_id] > pedictions_log_probs[-1]
                ):
                    pedictions[-1] = tok_id
                    pedictions_log_probs[-1] = logprobs_first_tok[tok_id]
        acc = sum([1 if gt[i] == pedictions[i] else 0 for i in range(len(gt))]) / len(
            gt
        )
        return acc


def mmlu_get_loss(model, dm: PlatypusModule, graph_string, use_vllm=True):
    # use gpu if available
    train_loss = 0
    if use_vllm:
        # use vllm
        model_hash = hashlib.sha256()
        model_hash.update(f"{graph_string}_{model.model.__class__}".encode())
        model = _LLMEngineMMLU(
            model,
            temp_path=f"{os.environ.get('MTTL_TEMP','/tmp/merged')}/{model_hash.hexdigest()}/",
        )
        dataloader: DataLoader = dm.val_dataloader(do_tokenize=False)
        acc = model.get_losses(
            dataloader,
            top_p=0.9,
            temperature=0.9,
            max_tokens=100,
            tokenizer=dm.tokenizer,
        )
        del model
        free_memory()
        return acc
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
    os.environ["MTTL_TEMP"] = "/home/v-oostapenko/mttl_out/models/merged/"

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

    dm = MMLUDataModule(config, for_generation=True)

    model_class = MultiExpertModel
    module = model_class(**vars(config), tokenizer=dm.tokenizer)

    main(s, dm=dm, model=module)
