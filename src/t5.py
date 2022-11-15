import logging
import re
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5ForConditionalGeneration

from adapters import Adapter, get_selector
from bart import SkilledModel, inform_layers
from utils import get_checkpoint_path, label_smoothed_nll_loss

EPS = 1e-12

logger = logging.getLogger(__name__)


class T5WithAdapter(T5ForConditionalGeneration, SkilledModel):
    def skilled_forward(
        self,
        input_ids,
        decoder_input_ids=None,
        task_ids=None,
        is_training=False,
        **kwargs
    ):
        # share the good news
        if task_ids is not None:
            inform_layers(self, task_ids)

        _decoder_input_ids = self.prepare_decoder_input_ids_from_labels(
            decoder_input_ids
        )
        outputs = self.forward(
            input_ids, decoder_input_ids=_decoder_input_ids, **kwargs
        )

        if is_training:
            loss, _ = label_smoothed_nll_loss(
                F.log_softmax(outputs.logits, dim=-1),
                decoder_input_ids,
                epsilon=0.1,
                ignore_index=self.config.pad_token_id,
            )
            return loss
        return outputs

    def generate(
        self,
        task_ids,
        task_embeds=None,
        defs_input_ids=None,
        defs_attn_mask=None,
        *args,
        **kwargs
    ):
        # share the good news
        inform_layers(self, task_ids)

        return super().generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, args):
        # loading back-compatibility
        if not hasattr(args, "n_splits"):
            args.n_splits = 1

        # First, get pretrained model des zinternet
        model = super().from_pretrained(args.model)

        if getattr(args, 'backbone_checkpoint', None):
            checkpoint = get_checkpoint_path(args.backbone_checkpoint)
            ckpt_ = torch.load(checkpoint)

            print("Loading backbone from a checkpoint: {}".format(checkpoint))
            for n, p in model.named_parameters():
                p.data.copy_(ckpt_["state_dict"][f"model.{n}"])

            # avoid reloading in the future
            args.backbone_checkpoint = None

        for p in model.parameters():
            p.requires_grad = args.finetune_full_model

        # Second, Build Adapter according to the specs
        if args.finegrained:
            replace_layers(model, nn.Linear, Adapter, "model", args)
        else:
            # use the same selector for all adapters
            adapter = partial(Adapter, selector=get_selector(args.selector)(args))
            replace_layers(model, nn.Linear, adapter, "model", args)

        return model


def replace_layers(model, old, new_module, parent_name, args):
    if args.finetune_full_model:
        return

    parent_names = ".*SelfAttention|.*EncDecAttention"

    for child_name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old, new_module, child_name, args)

        to_replace = {
            "lora": ["k", "v", "q", "o"],
            "ia3": ["k", "v", "o"],
        }

        if (
            re.fullmatch(parent_names, parent_name)
            and child_name in to_replace[args.processor]
        ):
            new = new_module(args, module)
            setattr(model, child_name, new)
