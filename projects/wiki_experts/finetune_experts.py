import os
import sys
import pytorch_lightning as pl
import glob

import copy
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.datamodule.mbpp_datamodule import MBPPDataConfig, MBPPDataModule
from mttl.datamodule.mmlu_data_module import MMLUDataConfig, MMLUDataModule

from mttl.models.modifiers.expert_containers.expert_library import (
    HFExpertLibrary,
    ExpertLibrary,
    LocalExpertLibrary,
    VirtualLocalLibrary,
    retry,
)

from mttl.callbacks import LiveCheckpointCallback

from mttl.models.monitors import get_monitors
from projects.wiki_experts.src.callbacks import DownstreamEvalCallback
from projects.wiki_experts.src.expert_model import MoETrainer, RoutedMultiExpertModel
from mttl.models.modifiers.expert_containers.module_graph import load_expert, Expert
from projects.wiki_experts.src.evolution.retrievers import (
    RandomRetriever,
    SVDEmbeddingRetriever,
)


import torch
from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything

from projects.wiki_experts.utils import get_datamodule
from mttl.callbacks import NanoMMLUCallback, RougeCallback
from mttl.utils import (
    get_checkpoint_path,
    get_pl_loggers,
    setup_logging,
    logger,
)
from mttl.models.modifiers.expert_containers.library_transforms import (
    SVDEmbeddingTransform,
    SVDEmbeddingTransformConfig,
)
from typing import Callable
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.train_experts_main import create_transfer_matrix
from projects.wiki_experts.src.callbacks import RougeLCallback


FINETUNE_FUNCTIONS: dict[str, Callable] = {}


@retry(max_retries=5, wait_seconds=60)
def svd_transform_with_retry(svd_embedder, expert_lib, upload_to_hf=True, force=True):
    return svd_embedder.transform(expert_lib, upload_to_hf=upload_to_hf, force=force)


def register_finetune_func(name):
    def decorator(func):
        if name not in FINETUNE_FUNCTIONS:
            FINETUNE_FUNCTIONS[name] = func
        else:
            raise ValueError(f"Duplicate name {name} in finetune functions")
        return func

    return decorator


def get_task_expert(task, library):
    if task not in library.tasks:
        raise ValueError(f"Task {task} not found in repository.")
    task_experts = []
    for name, metadata in library.data.items():
        if metadata.expert_deleted:
            continue
        if metadata.expert_task_name == task:
            task_experts.append(name)

    assert len(task_experts) == 1, f"Found {len(task_experts)} experts for task {task}"
    return library[task_experts[0]]


def load_lora_expert_in_expert_trainer(module: ExpertTrainer, expert: Expert):
    keys = module.model.load_state_dict(expert.expert_weights, strict=False)
    assert sum(["lora" in k for k in keys.missing_keys]) == 0, "Some keys are missing"


def load_expert_from_checkpoint(checkpoint):
    ckpt = torch.load(checkpoint)
    if "expert_dumps" in ckpt:
        expert_dumps = ckpt["expert_dumps"]
        expert: Expert = Expert.fromdict(expert_dumps)
    else:
        expert: Expert = load_expert(checkpoint)
    return expert


def prepare_expert_lib(args: ExpertConfig, lib_location) -> LocalExpertLibrary:
    library = LocalExpertLibrary.create_from_remote(
        HFExpertLibrary(args.hf_lib_id), destination=lib_location
    )
    if args.remove_experts is not None:
        remove_experts = args.remove_experts.split(",")
        for expert_name in remove_experts:
            logger.info(f"Removing expert {expert_name}")
            library.remove_expert(expert_name, soft_delete=False)
    return library


def create_mean_expert(args: ExpertConfig, library: ExpertLibrary = None) -> Expert:
    args_copy = copy.deepcopy(args)
    args_copy.router_selector = "uniform"
    lib_location = None
    if library is None:
        lib_location = f"/tmp/{args.hf_lib_id}"
        os.makedirs(lib_location, exist_ok=True)
        library = prepare_expert_lib(args, lib_location)
    module = RoutedMultiExpertModel(**vars(args_copy), device_map="auto")
    module.load_from_module_dict(library)
    mean_expert: Expert = module.to_expert()
    if lib_location is not None:
        shutil.rmtree(lib_location, ignore_errors=True)
    return mean_expert


def retrieve(args: ExpertConfig, task, k, retrieve_with="random"):
    if retrieve_with == "random":
        k = args.sk
        retriever = RandomRetriever(args, sk=k)

        lib_location = f"/tmp/{args.hf_lib_id}"
        os.makedirs(lib_location, exist_ok=True)
        library = prepare_expert_lib(args, lib_location)
        library: VirtualLocalLibrary = retriever.transform(
            library, current_task=args.finetune_task_name
        )
    elif retrieve_with == "svdemb":
        k = args.sk
        assert args.hf_repo_query is not None, "Please specify hf_repo_query"
        query_library = HFExpertLibrary(args.hf_repo_query)
        task = args.finetune_task_name
        query_expert: Expert = get_task_expert(task, query_library)

        retriever = SVDEmbeddingRetriever(args, sk=k)

        lib_location = f"/tmp/{args.hf_lib_id}"
        os.makedirs(lib_location, exist_ok=True)
        library = prepare_expert_lib(args, lib_location)
        if query_expert in library:
            library.remove_expert(query_expert.name)
        library.add_expert(query_expert)
        ###########################################################################
        # redo SVD with the query expert included
        if "neo" in args.hf_lib_id:
            sparsity_threshold = 0.5
        elif "phi" in args.hf_lib_id:
            sparsity_threshold = 0.5
        else:
            raise ValueError("'Neo' nor 'phi' in hf_lib_id,sparsity_threshold not set")

        logger.info(f"!!!!!!! Using sparsity threshold {sparsity_threshold}")
        svd_embedder = SVDEmbeddingTransform(
            SVDEmbeddingTransformConfig(sparsity_threshold=sparsity_threshold),
            random_state=42,
        )
        svd_transform_with_retry(svd_embedder, library, upload_to_hf=True, force=True)
        ###########################################################################
        library: VirtualLocalLibrary = retriever.transform(
            library, current_task=args.finetune_task_name, task_expert=query_expert
        )
    else:
        raise ValueError(f"Unknown retriever {retrieve_with}")
    return library


@register_finetune_func("lib_mu")
def finetune_lib_mu(args: ExpertConfig, dm):
    """
    1. Averages the library to a single expert
    2. Fine-tunes this expert on the downstream task
    """
    mean_expert: Expert = create_mean_expert(args)

    module = ExpertTrainer(**vars(args), device_map="auto")
    module.to("cuda")
    load_lora_expert_in_expert_trainer(module, mean_expert)

    checkpoint = train_module(args, module, dm)
    return load_expert_from_checkpoint(checkpoint), checkpoint


@register_finetune_func("lib_mu_randretr")
def finetune_lib_mu_with_rand_retrieval(args: ExpertConfig, dm):
    """
    1. Retrieves randomly args.sk experts from the library
    2. Averages the library to a single expert
    3. Fine-tunes this expert on the downstream task
    """
    library = retrieve(args, args.finetune_task_name, args.sk, retrieve_with="random")
    assert (
        len(library) == args.sk
    ), f"Retrieved {len(library)} experts, expected {args.sk}"

    mean_expert: Expert = create_mean_expert(args, library)

    module = ExpertTrainer(**vars(args), device_map="auto")
    module.to("cuda")
    load_lora_expert_in_expert_trainer(module, mean_expert)

    checkpoint = train_module(args, module, dm)
    return load_expert_from_checkpoint(checkpoint), checkpoint


@register_finetune_func("lib_mu_svdretr")
def finetune_lib_mu_with_svd_retrieval(args: ExpertConfig, dm):
    """
    1. Retrieves randomly args.sk experts from the library using SVD embeddings
    2. Averages the library to a single expert
    3. Fine-tunes this expert on the downstream task
    """
    library = retrieve(args, args.finetune_task_name, args.sk, retrieve_with="svdemb")
    assert (
        len(library) == args.sk
    ), f"Retrieved {len(library)} experts, expected {args.sk}"

    mean_expert: Expert = create_mean_expert(args, library)

    module = ExpertTrainer(**vars(args), device_map="auto")
    module.to("cuda")
    load_lora_expert_in_expert_trainer(module, mean_expert)

    checkpoint = train_module(args, module, dm)
    return load_expert_from_checkpoint(checkpoint), checkpoint


@register_finetune_func("polylib_full")
def finetune_polylib_full(args: ExpertConfig, dm):
    """
    Tunes selector and experts on downstream task.

    Returns the resulting expert.
    """

    args.trainable_param_names = (
        args.trainable_param_names
        + "|.*module_logits.*|.*selector.*"  # adds selector params to trainable params
    )
    # args.router_selector = "poly_router"
    assert args.router_selector is not None
    module = MoETrainer(**vars(args), device_map="auto")

    for n, p in module.named_parameters():
        if "selector" in n:
            assert p.requires_grad

    module.to("cuda")
    checkpoint = train_module(args, module, dm)
    return load_expert_from_checkpoint(checkpoint), checkpoint


@register_finetune_func("polylib_uniform")
def finetune_polylib_full(args: ExpertConfig, dm):
    args.router_selector = "uniform"
    module = MoETrainer(**vars(args), device_map="auto")

    module.to("cuda")
    checkpoint = train_module(args, module, dm)
    return load_expert_from_checkpoint(checkpoint), checkpoint


@register_finetune_func("polylib_selector")
def finetune_polylib_sel(args: ExpertConfig, dm):
    """
    Only trains the selector on the downstream task.
    """

    args.trainable_param_names = "|.*module_logits.*|.*selector.*"
    # args.router_selector = "poly_router"
    assert args.router_selector is not None

    module = MoETrainer(**vars(args), device_map="auto")
    for n, p in module.named_parameters():
        if "selector" in n:
            assert p.requires_grad
    module.to("cuda")
    checkpoint = train_module(args, module, dm)
    return load_expert_from_checkpoint(checkpoint), checkpoint


@register_finetune_func("polylib_full_randretr")
def finetune_polylib_full_with_rand_retrieval(args: ExpertConfig, dm):
    """
    Like polylib_full, but here we perform random expert selection before training.
    """
    library = retrieve(args, args.finetune_task_name, args.sk, retrieve_with="random")
    assert (
        len(library) == args.sk
    ), f"Retrieved {len(library)} experts, expected {args.sk}"

    # assert args.router_selector == "poly_router"
    assert args.router_selector is not None
    module = MoETrainer(**vars(args), device_map="auto", expert_library=library)

    for n, p in module.named_parameters():
        if "selector" in n:
            assert p.requires_grad

    module.to("cuda")
    checkpoint = train_module(args, module, dm)
    return load_expert_from_checkpoint(checkpoint), checkpoint


@register_finetune_func("private")
def finetune_private(args: ExpertConfig, dm):
    """
    Just train an expert from scratch
    """
    module = ExpertTrainer(**vars(args), device_map="auto")
    module.to("cuda")
    checkpoint = train_module(args, module, dm)
    return load_expert_from_checkpoint(checkpoint), checkpoint


@register_finetune_func("polylib_full_svdretr")
def finetune_polylib_full_with_svd_retrieval(args: ExpertConfig, dm):
    """
    Like polylib_full, but here we perform expert selection with SVD embeddings before training.
    """
    library = retrieve(args, args.finetune_task_name, args.sk, retrieve_with="svdemb")
    assert (
        len(library) == args.sk
    ), f"Retrieved {len(library)} experts, expected {args.sk}"

    # args.router_selector = "poly_router"
    assert args.router_selector is not None
    module = MoETrainer(**vars(args), device_map="auto", expert_library=library)

    for n, p in module.named_parameters():
        if "selector" in n:
            assert p.requires_grad

    module.to("cuda")
    checkpoint = train_module(args, module, dm)
    return load_expert_from_checkpoint(checkpoint), checkpoint


def run_multitask(args: ExpertConfig):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # select dataloader
    dm = get_datamodule(args)
    args.n_tasks = len(dm._task_names)

    if args.checkpoint is not None:
        # Passing a checkpoint assumes the use of `ExpertTrainer`
        # e.g. for poly-μ and MHR-μ
        ckpt_path = get_checkpoint_path(args.checkpoint)
        expert = load_expert(ckpt_path)
        module = ExpertTrainer(**vars(expert.training_config))

        ckpt = torch.load(ckpt_path)
        result = module.load_state_dict(ckpt["state_dict"], strict=False)
        assert len(result.unexpected_keys) == 0, result.unexpected_keys

        # For Poly and MHR, apply potential averaging, or resizing
        if args.finetune_type and args.finetune_type == "MuZ":
            module.model.switch_selector_to_average()
        elif expert.training_config.model_modifier == "poly":
            module.model.resize_module_logits(1)
        checkpoint = train_module(args, module, dm)
        if args.create_transfer_matrix:
            create_transfer_matrix(args, checkpoint)

    elif args.hf_lib_id is not None:
        # fine-tuning with expert library
        assert args.finetune_regime in FINETUNE_FUNCTIONS
        expert, checkpoint = FINETUNE_FUNCTIONS[args.finetune_regime](args, dm)

        if args.create_transfer_matrix:
            if "polylib" in args.finetune_regime:
                create_transfer_matrix(args, expert)
            else:
                create_transfer_matrix(args, checkpoint)

        shutil.rmtree(f"/tmp/{args.hf_lib_id}", ignore_errors=True)
        # can load expert to hf lib optionally here
    else:
        raise ValueError("please specify a library, or a checkpoint")


def train_module(args: ExpertConfig, module: ExpertTrainer, dm):
    loggers = get_pl_loggers(args)
    # get metric monitors for models
    callbacks = get_monitors(args)
    if "mbpp" in args.dataset:
        monitor = "downstream/mbpp"
        mode = "max"
    else:
        monitor = "val/loss"
        mode = "min"

    if "rouge" in args.es_metric:  # early stop on Rouge
        from projects.wiki_experts.src.callbacks import RougeLCallback

        dm_for_gen = get_datamodule(args, for_generation=True)
        checkpoint_callback = RougeLCallback(
            datamodule=dm_for_gen,
            output_dir=args.output_dir,
            eval_every_opt_step=args.eval_every,
            name=f"rougeL_val",
            split="val",
            checkpoint_oracle=True,
        )
        val_check_interval = 0.2
    else:
        checkpoint_callback = LiveCheckpointCallback(
            dirpath=args.output_dir,
            monitor=monitor,
            save_last=True,
            mode=mode,
        )
        val_check_interval = args.eval_every
        if val_check_interval == -1 or val_check_interval is None:
            val_check_interval = None
        else:
            val_check_interval = args.gradient_accumulation_steps * args.eval_every
            if val_check_interval > len(dm.train_dataloader()):
                val_check_interval = len(dm.train_dataloader())
            elif val_check_interval > args.total_steps and args.total_steps != -1:
                val_check_interval = args.total_steps

    callbacks.append(checkpoint_callback)
    eval_callback = None

    if args.pipeline_eval_tasks:
        if args.pipeline_eval_tasks == "all":
            args.pipeline_eval_tasks = "arc_challenge,arc_easy,boolq,hellaswag,humaneval,mbpp,openbookqa,piqa,bbh-fast,winogrande"

        eval_callback = DownstreamEvalCallback(args)
        callbacks.append(eval_callback)
    else:
        logger.warn(
            "Deactivating downstream eval callback as it is not enabled in the config. Please set `pipeline_eval_tasks`."
        )

    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        logger=loggers,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        strategy=args.compute_strategy if args.compute_strategy else "auto",
        callbacks=callbacks,
        enable_checkpointing=False,
        log_every_n_steps=args.gradient_accumulation_steps,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        val_check_interval=val_check_interval,
    )

    # initial validation only for a bunch of datasets... ?
    trainer.validate(module, dm)
    trainer.fit(module, dm)

    checkpoint = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    module.load_state_dict(torch.load(checkpoint)["state_dict"])
    trainer.test(module, dm)
    return checkpoint


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
