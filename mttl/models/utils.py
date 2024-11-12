import os
import re
from collections import defaultdict, deque

import prettytable
import torch

from mttl.logging import logger


def compute_loglike_loss(logits, labels, reduction="none"):
    bs = logits.size(0)
    vocab_size = logits.size(-1)
    labels = labels.squeeze(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    # reshape back
    if reduction == "none":
        loss = loss.view((bs, -1))
        # mean only non-zero
        non_zero_loss = (loss != 0).sum(dim=-1)
        non_zero_loss[non_zero_loss == 0] = 1
        loss = loss.sum(dim=-1) / non_zero_loss
    return loss


def transfer_batch_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def get_global_batch_size(batch_size, accumulation_steps):
    """Computes the global batch size."""
    try:
        world_size = torch.distributed.get_world_size()
    except:
        world_size = 1
    global_bs = batch_size * world_size * accumulation_steps
    return global_bs


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )

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

        # FIX for enabling gradient of the auxiliary loss
        # enable gradient checkpointing for memory efficiency
        from functools import partial

        notfailing_checkpoint = partial(
            torch.utils.checkpoint.checkpoint, use_reentrant=False
        )
        torch.utils.checkpoint.checkpoint = notfailing_checkpoint
        model.gradient_checkpointing_enable()
        # FIX for enabling gradient of the auxiliary loss
    return model


def model_loader_helper(
    model_name,
    device_map="auto",
    bf16=True,
    fp16=False,
    load_in_4bit=False,
    load_in_8bit=False,
    attn_implementation=None,
):
    if load_in_4bit and load_in_8bit:
        raise ValueError("Specify either 'load_in_4bit' or 'load_in_8bit' or neither.")

    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        BitsAndBytesConfig,
        LlamaForCausalLM,
        PreTrainedModel,
    )

    if load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    logger.info(f"Attention Implementation: {attn_implementation}")

    # set dtype
    if bf16:
        torch_dtype = torch.bfloat16
    elif fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if isinstance(model_name, PreTrainedModel):
        return model_name.train()

    if "phi-2" == model_name:
        # local phi-2 version. use `microsoft/phi-2 for the official hf version`
        if "PHI_PATH" not in os.environ:
            raise ValueError("PHI_PATH is not set in the environment variables.")

        model_name = os.environ["PHI_PATH"]
        logger.info(f"Loading phi-2 model from {os.environ['PHI_PATH']}")
    else:
        model_object = None
        exception = None
        for klass in [AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
            try:
                model_object = klass.from_pretrained(
                    model_name,
                    device_map=device_map,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    quantization_config=bnb_config,
                    torch_dtype=torch_dtype,
                )
                break
            except Exception as e:
                logger.warning(f"Couldn't load {model_name}! Exception: {e}")
                exception = e
                continue
        if model_object is None:
            raise ValueError(f"Couldn't load {model_name}!")

    if bnb_config is not None:
        model_object = prepare_model_for_kbit_training(model_object)

    logger.info("Freezing parameters of the base model by default.")
    for n, p in model_object.named_parameters():
        p.requires_grad = False

    return model_object.train()


# https://github.com/facebookresearch/dino/blob/main/utils.py
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]


class MetricLogger(object):
    def __init__(self):
        self.meters = defaultdict(SmoothedValue)

    def update(self, prefix=None, value_dict={}):
        prefix = "" if prefix is None else f"{prefix}/"
        for k, v in value_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[f"{prefix}{k}"].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def pretty_table(self, match_on=".*"):
        table = prettytable.PrettyTable()
        table.field_names = ["Name", "Value"]
        for name, meter in self.meters.items():
            if re.match(match_on, name):
                table.add_row([name, f"{meter.avg:.5f}"])

        return str(table)

    def __len__(self):
        return len(self.meters)
