import copy
import os

import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from args import get_args_parser
from callbacks import ProgressCallback
from data_module import FinetuneDataModule, NIFinetuneDataModule
from pl_model import Finetuner
from utils import get_checkpoint_path

# When loading a checkpoint for evaluation, which args from old checkpoint
# should overwrite the incoming arguments ?
ARGS_TO_OVERWRITE = [
    "combinor",
    "dataset",
    "finegrained",
    "lora_rank",
    "max_input_length",
    "max_output_length",
    "model",
    "n_skills",
    "n_splits",
    "n_tasks",
    "processor",
    "selector",
    "use_task_descriptions",
    "use_precomputed_task_embeddings",
    "num_pos_examples",
]


def finetune(args):
    seed_everything(args.seed, workers=True)

    # build the pretrained model
    ckpt_path = get_checkpoint_path(args.checkpoint)
    ckpt_args = torch.load(ckpt_path)['hyper_parameters']
    args.old_exp_name = ckpt_args['exp_name']

    for arg_name in ARGS_TO_OVERWRITE:
        if arg_name in ckpt_args:
            print("Overwriting", arg_name, "=", ckpt_args[arg_name])
            setattr(args, arg_name, ckpt_args[arg_name])

    kwargs = copy.deepcopy(vars(args))
    kwargs.pop('checkpoint')
    # these change for xfit, problematic for mlflow logger
    kwargs.pop('prefix')
    kwargs.pop('train_file')
    kwargs.pop('test_file')
    kwargs.pop('dev_file')
    module = Finetuner.load_from_checkpoint(ckpt_path, **kwargs, strict=False)

    # allocate new module logits for the new task
    if "polytropon" in module.hparams.selector:
        if args.finetune_switch_to_avg_modules:
            module.model.switch_selector_to_average(module.hparams)
        else:
            # resize to accomodate for new task
            module.model.resize_module_logits(1)

    if args.finetune_fix_skills:
        for n, p in module.named_parameters():
            if 'module_logits' not in n:
                p.requires_grad = False

    for n, p in module.named_parameters():
        if p.requires_grad:
            print("Finetuning: ", n)

    # data
    if args.dataset == "xfit":
        dm = FinetuneDataModule(args)
    elif args.dataset == "ni":
        dm = NIFinetuneDataModule(args)
    else:
        raise NotImplementedError()

    wandb_logger = WandbLogger(
        project="polytropon-ni",
        name=args.exp_name,
    )
    wandb_logger.experiment.save("*.py")

    # model checkpointing
    callback = ModelCheckpoint(
        monitor="val/metric_perf",
        filename="mbart-{epoch:02d}-{val/metric_perf:.2f}",
        save_top_k=1,
        mode="max",
    )

    trainer = Trainer(
        gpus=-1,
        accelerator="gpu",
        logger=wandb_logger,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        max_steps=args.total_steps,
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=10,
        check_val_every_n_epoch=10 if args.dataset == 'xfit' else 1,
        strategy="ddp_find_unused_parameters_false",
        limit_val_batches=1.0,
        limit_train_batches=1.0,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        callbacks=[ProgressCallback(), callback],
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )
    trainer.fit(module, dm)
    results = trainer.test(module, dm, ckpt_path="best")
    return results, callback.best_model_score.item(), dm.val_wrapper.metric


def finetune_ni(args):
    import pickle

    all_results = []
    columns = ['test_performance', 'test_exact_match', 'dev_performance', 'metric', 'seed']
    df = pd.DataFrame(columns=columns)

    for seed in [1, 2, 3]:
        args.seed = seed
        results, dev_perf, metric = finetune(args)
        all_results.append(results)
        test_em_perf, test_perf = (
            results[0][f"test/em_perf"],
            results[0][f"test/metric_perf"],
        )

        values = [test_perf, test_em_perf, dev_perf, metric, seed]
        print(
            "seed={},test_performance={}".format(seed, test_perf)
        )
        df.loc[len(df.index)] = values

    # whatever
    print(all_results)

    # dump results
    df.to_csv(os.path.join(args.output_dir, "result.csv"))
    with open(os.path.join(args.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(all_results, f)


def finetune_xfit(args):
    args.task_name = args.finetune_task_name
    args.task_dir = os.path.join(args.train_dir, args.task_name)
    files = sorted(os.listdir(args.task_dir))

    prefixes = []
    for filename in files:
        if not filename.endswith(".tsv"):
            continue

        prefix = "_".join(filename.split("_")[:-1])
        if prefix not in prefixes:
            prefixes.append(prefix)

    args.n_subtasks = len(prefixes)

    print("Fine-tuning the following samples: {}".format(prefixes))
    columns = ['prefix', 'test_performance', 'test_exact_match', 'dev_performance', 'metric', 'seed']

    df = pd.DataFrame(columns=columns)

    for prefix in prefixes:
        args.prefix = prefix
        args.train_file = os.path.join(args.task_dir, prefix + "_train.tsv")
        args.dev_file = os.path.join(args.task_dir, prefix + "_dev.tsv")
        args.test_file = os.path.join(args.task_dir, prefix + "_test.tsv")

        print(f"Running ... prefix={prefix}")

        for seed in [args.seed]:
            args.seed = seed
            results, dev_perf, metric = finetune(args)
            test_em_perf, test_perf = (
                results[0][f"test/em_perf"],
                results[0][f"test/metric_perf"],
            )

            values = [prefix, test_perf, test_em_perf, dev_perf, metric, seed]
            print(
                "seed={},prefix={},test_performance={}".format(seed, prefix, test_perf)
            )
            df.loc[len(df.index)] = values
    try:
        df.to_csv(os.path.join(args.output_dir, args.task, "result.csv"))
    except:
        df.to_csv(os.path.join(args.output_dir, "result.csv"))


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    if args.dataset == 'xfit':
        finetune_xfit(args)
    elif args.dataset == 'ni':
        finetune_ni(args)
