import glob
import json
import os   
import logging

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Optional, Sequence
import hashlib
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.autograd.function import Function

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import bitsandbytes as bnb
import transformers
import peft
from peft.tuners.lora import LoraLayer

logger = logging.getLogger(__name__)


def hash_example(example):
    return hashlib.md5(example.encode("utf-8")).hexdigest()


def template_to_string(template):
    return template.jinja + (
        (" answer_choices: " + template.answer_choices)
        if template.answer_choices
        else ""
    )


def label_smoothed_nll_loss(
    lprobs, target, epsilon=0.1, ignore_index=-100, reduction="mean"
):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduction == "mean":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = epsilon / lprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

        # NOTE (Lucas): the original code does not divide by the batch size. Not great
        loss, nll_loss = loss / lprobs.size(0), nll_loss / lprobs.size(0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        eps_i = epsilon / lprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def get_tasks_list(filename, split_name):
    with open(filename, "r") as fin:
        split_dict = json.load(fin)
    return split_dict[split_name]


def get_ni_tasks_from_file(filename):
    with open(filename, "r") as f:
        tasks = f.readlines()
        tasks = [task.strip() for task in tasks]
    task2id = {task: idx for idx, task in enumerate(tasks)}
    return tasks, task2id


def get_example_to_ids(filename):
    import pickle

    with open(filename, "rb") as f:
        package = pickle.load(f)
    return package


def average_dicts(list_of_dicts):
    out = list_of_dicts[0]
    for item in list_of_dicts[1:]:
        assert len(item) == len(out)
        for k, v in item.items():
            out[k] += v

    return {k: v / len(list_of_dicts) for (k, v) in out.items()}


class CustomModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_model_score = None

    def _update_best_and_save(
        self,
        current: Tensor,
        trainer: "pl.Trainer",
        monitor_candidates: Dict[str, Tensor],
    ) -> None:
        """First remove checkpoint, THEN save it."""
        import os

        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, Tensor) and torch.isnan(current):
            current = torch.tensor(
                float("inf" if self.mode == "min" else "-inf"), device=current.device
            )

        filepath = self._get_metric_interpolated_filepath_name(
            monitor_candidates, trainer, del_filepath
        )

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )

        if del_filepath is not None and filepath != del_filepath:
            print(f"Removing checkpoint... {del_filepath}")
            trainer.strategy.remove_checkpoint(del_filepath)

        print(f"Saving checkpoint... {filepath}")
        self._save_checkpoint(trainer, filepath)
        os.system("df")
        os.system(f"ls -al {filepath}")

        self.last_model_score = current


def get_mlf_logger():
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    class MLFlowLoggerCustom(pl.loggers.MLFlowLogger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @rank_zero_only
        def log_hyperparams(self, *args, **kwargs) -> None:
            try:
                super().log_hyperparams(*args, **kwargs)
            except:
                pass

    try:
        from azureml.core.run import Run

        run = Run.get_context()
        mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()
        mlf_logger = MLFlowLoggerCustom(
            experiment_name=run.experiment.name, tracking_uri=mlflow_url
        )
        mlf_logger._run_id = run.id
    except:
        mlf_logger = None
    return mlf_logger


def get_checkpoint_path(path, step=None):
    if path.endswith(".ckpt") or path.endswith(".pt"):
        return path

    # use glob to avoid explicitly writing out long paths
    match = glob.glob(f"{path}/*.ckpt", recursive=True)

    if len(match) > 1:
        logger.warning(
            f"{len(match)} checkpoints found. "
            + "taking the one with the lowest val loss"
        )
        losses = []
        for x in match:
            if "loss" in x:
                loss = float(x.split("loss=")[-1].split(".ckpt")[0])
            elif "zero_shot_perf" in x:
                loss = -float(x.split("zero_shot_perf=")[-1].split(".ckpt")[0])
            else:
                continue
            losses.append(loss)
        idx = np.argmin(losses) if losses else 0
        path = match[idx]
    elif len(match) == 0:
        match = glob.glob(f"{path}/*step*.pt", recursive=True)
        if len(match) > 1:
            logger.warning(
                f"{len(match)} checkpoints found. "
                + "taking the one with the lowest val loss"
            )
            found = False
            for m in match:
                # take the one with the specified step
                if str(step) in m:
                    path = m
                    found = True
                    break
            if not found and step is None:
                # global_stepX.pt, take the one with the highest step
                idx = np.argmax(
                    [float(x.split("step")[-1].split(".pt")[0]) for x in match]
                )
                path = match[idx]
        elif len(match) == 0:
            raise FileNotFoundError(f"{path} had no `.ckpt` nor `.pt` files")
        else:
            path = match[0]
    else:
        path = match[0]

    print("Found checkpoint", path)
    return path


class MemEfficientLoRA(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, A, B, skills):
        ctx.save_for_backward(input, A, B, skills)

        bs, L, I = input.size()
        bs, n_skills = skills.size()
        n_skills, I, R = A.size()
        n_skills, R, O = B.size()

        output = torch.einsum("BLI,BS,SIR,SRO->BLO", (input, skills, A, B))

        return output

    @staticmethod
    def backward(ctx, O_grad):

        bs, L, O = O_grad.size()

        input, A, B, skills = ctx.saved_tensors

        bs, S, I = input.size()
        bs, n_skills = skills.size()
        n_skills, I, R = A.size()
        n_skills, R, O = B.size()

        # option A)
        W = torch.einsum("BS,SIR,SRO->BIO", (skills, A, B))
        I_grad = torch.einsum("BLO,BIO->BLI", (O_grad, W))

        # option B) [OOM]
        # I_grad = torch.einsum('BLO,BS,SIR,SRO->BLI', (O_grad, skills, A, B))

        W_grad = torch.einsum("BLO,BLI->BIO", (O_grad, input))

        tmp = torch.einsum("BIO,BS->SIO", (W_grad, skills))
        A_grad = torch.einsum("SIO,SRO->SIR", (tmp, B))
        B_grad = torch.einsum("SIO,SIR->SRO", (tmp, A))
        S_grad = torch.einsum("BIO,SIR,SRO->BS", (W_grad, A, B))

        return I_grad, A_grad, B_grad, S_grad


if __name__ == "__main__":
    B, L, S, I, O, R = 3, 5, 8, 3, 12, 4
    fn = MemEfficientLoRA.apply

    for i in range(10):
        input = torch.randn(B, L, I, dtype=torch.double).cuda()
        Am = torch.randn(S, I, R, dtype=torch.double).cuda()
        Bm = torch.randn(S, R, O, dtype=torch.double).cuda()
        skill = torch.randn(B, S, dtype=torch.double).cuda()
        idx1 = torch.multinomial(torch.ones(S, I * O).cuda(), num_samples=10)
        idx2 = torch.arange(S).repeat_interleave(10).cuda()
        idx = torch.stack([idx1.flatten(), idx2.flatten()])
        val = torch.randn(size=idx.shape[1:]).cuda().double()
        W = torch.sparse_coo_tensor(idx, val, (I * O, S)).coalesce()

        # coll = [input, Am, Bm, skill]
        coll = [input, W, skill]
        for x in coll:
            x.requires_grad = True

        res = torch.autograd.gradcheck(fn, coll, check_sparse_nnz=True)
        import pdb

        pdb.set_trace()
        print(res)



logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
_DEFAULT_VAL_LOCAL_RANK = (
    int(os.environ["LOCAL_RANK"]) 
    if "LOCAL_RANK" in os.environ else None
)

def _find_all_linear_names(bits: int, model, lm_head_name: str):
    """ Finds all the linear layer names in the model.

    This is to pass them as targets for LORA.

    Node that this doesn't work at all with GPT2 as it 
    uses 1D convs instead of linear layers.
    
    Model can possibly quantized, but it's not necessary.
    The lora targets need to be found, whether the model 
    is quantized or not.

    Args:
        bits:
            How many bits to use. 4, 8, 16, 32
        model:
            The possibly quantized huggingface model.

    """

    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (
            bnb.nn.Linear8bitLt 
            if bits == 8 
            else torch.nn.Linear
        )
    )

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(
                names[0] 
                if len(names) == 1 
                else names[-1]
            )

    if lm_head_name in lora_module_names:  # needed for 16-bit
        lora_module_names.remove(lm_head_name)

    return list(lora_module_names)


def _check_is_causal(model_name_or_path, trust_remote_code):
    """ Ensures that the model is causal.

    The QLora code is only with causal models.

    """
    try:
        config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)
    except OSError:
        logger.warning(
            "The model doesn't have a config.json. "
            "We assume that it's causal. Use --force_seq2seq to override."
        )
        return

    if vars(config).get("is_encoder_decoder", False):
        raise ValueError(
            "We haven't tested the code with encoder-decoder models yet. "
            f"Pass ignore_is_causal_check=True to "
            "`peft_qlora.from_pretrained` to ignore this error, "
            "but do so at your own risk."
        )


def _find_lm_head(model):
    """
    The original code tries to detect the lm head by checking for the presence
    of "lm_head" in the name of the module, which is again very flimsy. We try
    to be more general. We find the lm_head by assuming that objects created by 
    AutoModelForCausalLM have two modules as children, one 
    transformer.modeling_utils.PreTrainedModel, and the other is the lm_head.
    We make sure that lm_head is of a reasonable type for a lm_head.
    This is a lot more general.
    """
    children = [
        dict(name=name, module=module) 
        for name, module in model.named_children()
    ]
    assert len(children) == 2, len(children)

    pretrained_model = children[0]
    lm_head = children[1]

    assert isinstance(
        pretrained_model["module"],
        transformers.modeling_utils.PreTrainedModel), (
        type(pretrained_model))
    
    assert isinstance(lm_head["module"], torch.nn.Linear), type(lm_head)

    return lm_head

def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32
    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model

def from_pretrained(
    model_name_or_path: str,
    fp16: bool = False,
    bf16: bool = True,
    max_memory_MB: Optional[int] = None,
    cache_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    full_finetune: bool = False,
    gradient_checkpointing: bool =True,
    bits: int = 4,
    quant_type: str = "nf4",
    double_quant: bool = True,
    trust_remote_code: bool = False,
    use_auth_token: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    local_rank: Optional[int] = _DEFAULT_VAL_LOCAL_RANK,
    ignore_is_causal_check: bool = False,
    force_seq2seq: bool = False,
    add_lora_f=None,
):
    """Only public function of this library.

    Creates your model with QLora, using Peft and Bitsandbytes, 
    but also finding the all the possible linear layers instead of
    just k and v like in the regular Lora code.

    You can then use the model like you would a normal HuggingFace Peft Model.

    Very slightly modified from the original 
    qlora/qlora.get_accelerate_model to add the arguments and the defaults.
    
    Args:
        model_name_or_path: 
            Huggingface auto model from_pretrained name or path argument.
            No default.
        bf16: 
            Whether to use bf16.
            Default: True.
        fp16: 
            Whether to use fp16.
            Default: False.
        cache_dir: 
            Huggingface caching dir.
            Default: None.
        checkpoint_dir: 
            Huggingface checkpoint dir.
            Default: None.
        max_memory_MB: 
            Max gpu memory to use in Megabytes.
            Default: None.
        full_finetune: 
            Finetune the entire model without adapters.
            Default: False.
        gradient_checkpointing: 
            Use gradient checkpointing. You want to use this.
            Default: True.
        bits: 
            How many bits to use.
            Default: 4.
        quant_type: 
            Quantization data type to use. Should be one of `fp4` or `nf4`.
            Default: `nf4`.
        double_quant: 
            Compress the quantization statistics through double quantization.
            Default: True.
        trust_remote_code: 
            Enable unpickling of arbitrary code in AutoModelForCausalLM.from_pretrained.
            Default: False.
        use_auth_token: 
            Enables using Huggingface auth token from Git Credentials.
            Default: False.
        lora_r: 
            Lora R dimension.
            Default: 64.
        lora_alpha: 
            Lora alpha.
            Default: 16.
        lora_dropout: 
            Lora dropout.
            Default: 0.0.
        ignore_is_causal_check: 
            We added this. This is if you want to try using an encoder decoder. It's untested.
            Default: False.
        local_rank: 
            Local rank for distributed training. 
            Default: int(os.environ["LOCAL_RANK"]) if it exists.
    """

    # -------------------------------------------------------------------------
    # JulesGM: We added those checks, as well as `experimental`
    # support for encoder-decoder models. It should work out of the box,
    # but we added warnings & requiring turning of "ignore_is_causal_check"
    # because it's not in the original code.
    # -------------------------------------------------------------------------
    cls = transformers.AutoModelForCausalLM
    if force_seq2seq:
        cls = transformers.AutoModelForSeq2SeqLM
        logger.warning(
            "Seq2SeqLMs support is experimental. Use at your own risk."
        )
    elif ignore_is_causal_check:
        try:
            config = transformers.AutoConfig.from_pretrained(
                model_name_or_path, 
                trust_remote_code=trust_remote_code,
            )
            if vars(config).get("is_encoder_decoder", False):
                logger.warning(
                    "Encoder-decoder models are untested with this library.")
                cls = transformers.AutoModelForSeq2SeqLM
        except OSError:
            # This model doesn't have a config.json file, so we can't check
            # if it's an encoder-decoder model. 
            logger.warning(
                    "This model doesn't have a config.json file, "
                    "so we can't check if it's an encoder-decoder model. "
                    "Defaulting to causal. Use --force_seq2seq if you wanted "
                    "a causal model."
                )
    else:
        _check_is_causal(model_name_or_path, trust_remote_code)

    if fp16 and bf16:
        raise ValueError("Can't use both fp16 and bf16")

    assert bits in [4, 8, 16, 32], (
        f"bits must be one of 4, 8, 16, 32, got {bits = }")
    

    # -------------------------------------------------------------------------
    # JulesGM: We added support for max_memory = None, so it doesn't
    # automatically overflow to cpu offloading, which is slow and should not
    # happen silently. 
    # -------------------------------------------------------------------------
    n_gpus = torch.cuda.device_count()
    if max_memory_MB:
        max_memory = f"{max_memory_MB}MB"
        max_memory = {i: max_memory for i in range(n_gpus)}
    else:
        max_memory = None
    device_map = "auto"
    # if we are in a distributed setting, 
    # we need to set the device map and max memory per device
    if local_rank is not None:
        device_map = {"": local_rank}
        max_memory = (
            {"": max_memory[local_rank]} 
            if max_memory else None
        )
    # -------------------------------------------------------------------------


    if full_finetune:
        assert bits in [16, 32]

    logger.info(f"loading base model {model_name_or_path}...")
    compute_dtype = (
        torch.float16 
        if fp16 else (
            torch.bfloat16 
            if bf16 else torch.float32
        )
    )

    # JulesGM: This is identical to the original code.
    model = cls.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type=quant_type,
        ),
        torch_dtype=(
            torch.float32 
            if fp16 else (
                torch.bfloat16 
                if bf16 else torch.float32
            )
        ),
        trust_remote_code=trust_remote_code,
        use_auth_token=use_auth_token,
    )

    if compute_dtype == torch.float16 and bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, "
                "you can accelerate training with "
                "the argument --bf16"
            )
            print("=" * 80)

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = (
        torch.float32 
        if fp16 else (
            torch.bfloat16 
            if bf16 else torch.float32
        )
    )

    if not full_finetune:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()


    # -------------------------------------------------------------------------
    # JulesGM:
    # The original code modifies the lm_head by the name specific to type of
    # model they were fine-tuning, which is really very flimsy.
    #
    # We try to be more general.
    # -------------------------------------------------------------------------
    lm_head = _find_lm_head(model)
    # -------------------------------------------------------------------------

    # if not full_finetune:
    #     if checkpoint_dir is not None:
    #         logger.info("Loading adapters from checkpoint.")
    #         model = peft.PeftModel.from_pretrained(
    #             model, 
    #             os.path.join(checkpoint_dir, "adapter_model"), 
    #             is_trainable=True,
    #         )
    #     else:
    #         if add_lora_f is None:
            
    #             logger.info(f"Adding LoRA modules.")
    #             modules = _find_all_linear_names(
    #                 bits=bits, 
    #                 model=model, 
    #                 lm_head_name=lm_head["name"],
    #             )
                
    #             assert modules, f"{modules = }, {bits = }"
                
    #             config = peft.LoraConfig(
    #                 r=lora_r,
    #                 lora_alpha=lora_alpha,
    #                 target_modules=modules,
    #                 lora_dropout=lora_dropout,
    #                 bias="none",
    #             task_type="CAUSAL_LM",)
            
    #             model = peft.get_peft_model(model, config)
    #         else:
    #             model=add_lora_f(transformer=model)

    fp32_weights = []
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)

        # -------------------------------------------------------------------------
        # JulesGM:
        # The original code finds layer norms by the presence of "name"
        # in the names of the modules, which is again very flimsy.
        # We try to be more general by checking the type.
        # -------------------------------------------------------------------------
        if isinstance(
                module,   
                torch.nn.modules.normalization.LayerNorm
        ) or "LlamaRMSNorm" in str(module.__class__):  
            # -------------------------------------------------------------------------
            # JulesGM - FIX
            # The original code doesn't cast the layer norms to bfloat16, but to float32,
            # but that just didn't run for me at all.
            #
            # The idea from the rest of the code is to cast
            # non low bytes layers to bfloat16 in bf16 mode, and to float32 in fp16 mode.
            # So we changed it to cast layer norm layers to bfloat16 in bf16 mode, and left 
            # it to float32 in in other modes, and it works.
            #
            # This is the only somewhat significant change to the original code, but feels pretty
            # reasonable, and the model trains perfectly fine, and doesn't work otherwise.
            if bf16:
                module = module.to(torch.bfloat16)
            # -------------------------------------------------------------------------

            else:
                module = module.to(torch.float32)

        # -------------------------------------------------------------------------
        # JulesGM:
        # The original code tries to find the embedding by looking at "embed"
        # in the name of the module, which is again very flimsy. We try to be more
        # general by checking the type.
        # 
        # We also use our more general method of detecting the lm_head
        # -------------------------------------------------------------------------
        if (
            module is lm_head["module"] or
            isinstance(module, torch.nn.Embedding)  
        ):
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

        if bf16:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    fp32_weights.append((name, module.weight.dtype, type(module)))

    assert not bf16 or not fp32_weights, (
        f"Found fp32 weights in {fp32_weights}. "
    )

    return model
