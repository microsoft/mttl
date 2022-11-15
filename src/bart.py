import logging
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from adapters import Adapter, Selector, get_selector
from utils import get_checkpoint_path, label_smoothed_nll_loss

EPS = 1e-12

logger = logging.getLogger(__name__)


class SkilledModel:
    @property
    def selectors(self):
        added_selectors = set()
        for name, adapter in self.adapters:
            if adapter.selector not in added_selectors:
                added_selectors.add((name + ".selector", adapter.selector))
        return list(added_selectors)

    @property
    def adapters(self):
        adapters_ = []
        for n, m in self.named_modules():
            if isinstance(m, Adapter):
                adapters_.append((n, m))
        return adapters_

    def switch_selector_to_average(self, hparams):
        """Instead of learning new task logits, just use average.
        """
        from adapters import Average, Polytropon

        def _scan(module):
            for name, inner_mod in module.named_children():
                if len(list(inner_mod.children())) > 0:
                    _scan(inner_mod)

                if isinstance(inner_mod, Polytropon):
                    print("Replacing with average: ", name, "n_skills:", hparams.n_skills)
                    setattr(module, name, Average(hparams))
        _scan(self)

    def resize_module_logits(self, n_tasks):
        """Resizes the vector routing, in case of fine-tuning for example.
        """
        for name, selector in self.selectors:
            if hasattr(selector, 'module_logits'):
                print("Resizing module_logits of selector", name, "with", n_tasks, "tasks.")
                selector.module_logits.data = selector.init_module_logits(n_tasks)


class BartWithAdapter(BartForConditionalGeneration, SkilledModel):
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
            inform_layers(
                self,
                task_ids,
            )

        _decoder_input_ids = self.prepare_decoder_input_ids_from_labels(
            decoder_input_ids
        )
        outputs = self.model(input_ids, decoder_input_ids=_decoder_input_ids, **kwargs)

        lm_logits = F.linear(
            outputs[0], self.model.shared.weight, bias=self.final_logits_bias
        )

        masked_lm_loss = None

        if is_training:
            loss, _ = label_smoothed_nll_loss(
                F.log_softmax(lm_logits, dim=-1),
                decoder_input_ids,
                epsilon=0.1,
                ignore_index=self.config.pad_token_id,
            )
            return loss

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate(self, task_ids, task_embeds=None, *args, **kwargs):
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

        if args.finetune_full_model:
            return model

        for p in model.parameters():
            p.requires_grad = args.finetune_full_model

        # Second, Build Adapter according to the specs
        if args.finegrained:
            replace_layers(model, nn.Linear, Adapter, args)
        else:
            # use the same selector for all adapters
            adapter = partial(Adapter, selector=get_selector(args.selector)(args))
            replace_layers(model, nn.Linear, adapter, args)
        return model


def replace_layers(model, old, new_module, args):
    if args.finetune_full_model:
        return

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old, new_module, args)

        to_replace = {
            "lora": ["k_proj", "v_proj", "q_proj", "out_proj"],
            "ia3": ["k_proj", "v_proj", "fc2"],
        }

        if isinstance(module, old) and name in to_replace[args.processor]:
            new = new_module(args, module)
            setattr(model, name, new)


def inform_layers(model, tasks, **kwargs):
    for module in model.children():
        if len(list(module.children())) > 0:
            inform_layers(module, tasks, **kwargs)

        if isinstance(module, Selector):
            setattr(module, "tasks", tasks)

            for key, value in kwargs.items():
                setattr(module, key, value)
