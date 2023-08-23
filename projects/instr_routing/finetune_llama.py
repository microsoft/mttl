import os
import sys
import json
import torch
import wandb
import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.poly import get_selector
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from projects.instr_routing.models.clm import CLM
from mttl.callbacks import ProgressCallback
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.platypus_module import PlatypusModule
from mttl.datamodule.flan_module import FlanModule
from mttl.config import Config
from mttl.models.monitors import get_monitors
from mttl.utils import get_mlf_logger
from mttl.models.modify_model import modify_transformer
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from huggingface_hub import login
from peft import prepare_model_for_int8_training

# register models
import models.routing  # noqa: F401


##################################################
def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


from sklearn.feature_extraction.text import TfidfVectorizer


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def remove_non_serializable(d):
    """
    Recursively remove non-JSON serializable values from a dictionary.
    """
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            remove_non_serializable(v)
        elif not json.dumps(v, default=lambda x: None):
            del d[k]


class RoutingConfig(Config):
    def _set_defaults(self):
        super()._set_defaults()

        self.rank = 1
        self.load_dtype = "float32"
        self.prune_unused_loras = True
        self.init_b_random = False
        self.lora_dropout = 0
        self.lora_alpha = 16
        self.same_lora_init = 0
        self.load_in_8bit = False
        self.micro_batch_size = 4
        self.share_lora_at_attn = 0
        self.share_lora_a = False
        self.merge_A_B_seperately = True
        self.train_on_inputs = False
        self.padding_side = "right"
        self.adapter_modules = None
        self.router_selector_use_distances = False
        self.adapter_layers = 0  # llama adapter
        self.adapter_len = 0  # llama adapter
        self.use_4_bit_backbone = False
        self.wandb_project = None
        self.switch_to_average = 0

        self.router_weight_decay = None
        self.param_names_added_to_sd = ""  # define additional params that will be added to state dict additionally to the trainable ones.
        self.predict_cluster = None  # topic or skill
        self.dst_dir = None  # dir of jsonl dataset

        self.fast_dev_run = False
        self.fast_debug_run = False

        self.hf_token_hub = None
        self.eval_ds_limit = 1
        self.train_only_cluster = None
        self.validation_portion = 0.03
        self.per_cluster_test = False
        self.use_test_set = False  # wether to use examples marked as is_test = 1 in ClusterInfo as test set
        self.aux_mi_loss_factor = 1

        # XRouter
        self.xrouter_load_balancing = False
        self.xrouter_x_cond = True
        self.xrouting_option = 0  # only applies to x_router routing, depreciated
        self.xrouter_normalize_weights = False
        self.xrouter_normalize_input = False
        self.xrouter_reverse_kl = False
        self.xrouter_normal_innit = True
        self.xrouter_use_attn = False
        self.xrouter_sim_metric = "kl"
        self.xrouting_sep_teacher_student = False
        self.xrouter_init_scale = 0.02
        self.xrouter_x4target_detach = True
        self.xr4_option = None  # "switch" #, "default", "teacher_output"
        self.xrouter_x4_target = "posterior"  # , "prior" -- wich router logits to use for x4 at trainign time.

        self.superni_eval_batchsize = 2
        self.router_learning_rate = None
        self.eval_hellaswag = True
        self.eval_arc = True
        self.eval_truthfulqa = True
        self.eval_superni = True
        self.eval_mmlu = True
        self.eval_superni_use_outputs = False
        self.gen_alpaca_eval = False

        self.train_dir = os.getenv("AMLT_DATA_DIR", "~/data/")
        self.output_dir = os.getenv("AMLT_OUTPUT_DIR", "tmp/instruction_learning/")

    def post_init(self):
        # to reproduce setup in https://github.com/daanelson/alpaca-lora
        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size


def run_multitask(args):
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    print(os.path.dirname(os.path.realpath(__file__)))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    if args.example_to_ids_path:
        raise NotImplementedError()

    # select dataloader
    model_class = CLM
    if args.dataset == "alpaca":
        dm = AlpacaDataModule(args)
    elif args.dataset == "flan_v2":
        dm = FlanModule(args)
    elif args.dataset == "human":
        dm = FlanModule(args)
    elif args.dataset == "platypus":
        dm = PlatypusModule(args)
    else:
        raise NotImplementedError()

    if args.load_dtype == "float32":
        load_dtype = torch.float32
    elif args.load_dtype == "float16":
        load_dtype = torch.float16
    else:
        raise NotImplementedError()

    if "llama" in args.model:
        args.model_object = LlamaForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=args.load_in_8bit,  # this doesnt work right now with current implementatio of lora
            torch_dtype=load_dtype,
            device_map="auto",
        )
    else:
        args.model_object = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto"
        )

    if args.model_object.config.vocab_size != len(
        dm.tokenizer
    ):  # if adding [EOI] in LongForm dataset
        args.model_object.resize_token_embeddings(len(dm.tokenizer))
    if args.load_in_8bit:
        args.model_object = prepare_model_for_int8_training(args.model_object)

    model_object = modify_transformer(args.model_object, args)

    module = model_class(**vars(args), model_object=model_object, tokenizer=dm.tokenizer)
    del args.model_object

    if args.switch_to_average > 0:
        module.model.switch_selector_to_average(
            selector_to_replace=get_selector(args).__class__
        )

    # legit logging
    loggers = []
    if os.environ.get("WANDB_API_KEY") or args.wandb_project:
        project = (
            "alpaca_tuning_ncb" if args.wandb_project is None else args.wandb_project
        )
        project = os.environ.get("WANDB_PROJECT", project)
        project += f"_{args.dataset}"
        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=os.environ.get("AMLT_JOB_NAME", args.exp_name),  # , config=args_
        )
        wandb_logger.experiment.save("*.py")
        loggers.append(wandb_logger)
    else:
        wandb_logger = None

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    loggers.append(pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics"))

    kwargs = {"val_check_interval": args.eval_every} if args.eval_every else {}

    # get metric monitors for models
    callbacks = get_monitors(args)
    callbacks.append(ProgressCallback())

    monitor = "val/loss"
    mode = "min"

    model_name = args.model.replace("/", "_")
    # check if wandb run exists
    if wandb_logger:
        # get run id
        run_id = wandb_logger.experiment.id
        model_name += run_id
    exp_name = os.environ.get("AMLT_JOB_NAME", args.exp_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        monitor=monitor,
        filename=f"{model_name}" + f"_{exp_name}" + "-{" + monitor + ":.004f}",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,  # make checkpoints smaller
        mode=mode,
    )
    callbacks.append(checkpoint_callback)

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        logger=loggers,
        num_sanity_val_steps=5,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=20,
        strategy=args.compute_strategy if args.compute_strategy else "auto",
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        fast_dev_run=args.fast_dev_run,
        **kwargs,
    )

    trainer.fit(module, dm)

    ckpt_path = "best" if not args.fast_dev_run else None
    trainer.validate(dataloaders=dm, ckpt_path=ckpt_path)

    if args.use_test_set and not args.fast_dev_run:
        module.model.checkpoint_tested = "best"
        trainer.test(dataloaders=dm, ckpt_path=ckpt_path)

        module.model.checkpoint_tested = "last"
        trainer.test(dataloaders=dm, ckpt_path="last")

    path_best_model = trainer.checkpoint_callback.best_model_path
    path_last_model = trainer.checkpoint_callback.last_model_path

    print(f"Best model path: {path_best_model}")
    print(f"Last model path: {path_last_model}")

    # empty memory
    del (
        module,
        dm,
        trainer,
        loggers,
        callbacks,
        checkpoint_callback,
        wandb_logger,
        mlf_logger,
    )
    ds_limit = args.eval_ds_limit if not args.fast_debug_run else 0.05
    torch.cuda.empty_cache()

    if args.eval_superni:
        print("#" * 50)
        print("Evaluating on super NI")
        from projects.instr_routing.eval.ni.gen_ni_predictions import eval_superni

        rouge_L_super_ni = eval_superni(
            model_name="",
            batch_size=args.superni_eval_batchsize,
            out_prefix=f"{args.exp_name}",
            model_path=path_best_model,
            nshot=0,
            use_outputs=args.eval_superni_use_outputs,
            ds_limit=ds_limit,
        )
        if wandb.run is not None:
            wandb.log({"rouge_L_super_ni": rouge_L_super_ni})
        print("SuperNI RougeL: {:.2f}".format(rouge_L_super_ni))

    if args.eval_mmlu:
        from projects.instr_routing.eval.mmlu.run_mmlu_eval import eval_mlu

        print("#" * 50)
        print("Evaluating on MMLU")
        acc = eval_mlu(
            ntrain=5, model_name="", model_path=path_best_model, eval_batch_size=2
        )
        if wandb.run is not None:
            wandb.log({"mmlu_acc": acc})
        print("MMLU accuracy: {:.2f}".format(acc))

    if args.eval_arc:
        from projects.instr_routing.eval.lm_eval_harness.run_eval import eval_lm

        print("#" * 50)
        print("Evaluating on ARC")
        results_dict = eval_lm(
            model_path=path_best_model,
            model_name="",
            task="arc_challenge",
            batch_size=2,
            nshot=25,
            ds_limit=ds_limit,
        )
        if wandb.run is not None:
            wandb.log(results_dict)

        print("ARC results: {}".format(results_dict))

    if args.eval_truthfulqa:
        from projects.instr_routing.eval.lm_eval_harness.run_eval import eval_lm

        print("#" * 50)
        print("Evaluating on TruthfulQA")
        results_dict = eval_lm(
            model_path=path_best_model,
            model_name="",
            task="truthfulqa_mc",
            batch_size=2,
            nshot=0,
            ds_limit=ds_limit,
        )
        if wandb.run is not None:
            wandb.log(results_dict)
        print("TQA results: {}".format(results_dict))

    if args.eval_hellaswag:
        from projects.instr_routing.eval.lm_eval_harness.run_eval import eval_lm

        print("#" * 50)
        print("Evaluating on HellaSwag")
        results_dict = eval_lm(
            model_path=path_best_model,
            model_name="",
            task="hellaswag",
            batch_size=2,
            nshot=10,
            ds_limit=ds_limit,
        )
        if wandb.run is not None:
            wandb.log(results_dict)
        print("HSWAG results: {}".format(results_dict))


if __name__ == "__main__":
    args = RoutingConfig.parse()
    run_multitask(args)
