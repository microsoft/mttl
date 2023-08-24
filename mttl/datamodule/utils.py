import torch
from transformers import AutoTokenizer


def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    if config.model_family == 'gpt':
        tokenizer.padding_side = 'left'

    if tokenizer.pad_token_id is None:
        # no padding token, use EOS token instead!
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_inputs_for_gpt_family(batch: dict, tokenizer: AutoTokenizer):
    input_ids, labels = batch['input_ids'], batch['labels']
    eos_tokens = torch.full((labels.shape[0], 1), tokenizer.eos_token_id, device=labels.device)
    labels = torch.cat((labels, eos_tokens), dim=-1)
    padded_labels = torch.where((labels == -100), tokenizer.pad_token_id, labels)
    padded_input_ids = torch.full_like(input_ids, -100)
    batch['input_ids'] = torch.cat((input_ids, padded_labels), dim=-1)
    batch['labels'] = torch.cat((padded_input_ids, labels), dim=-1)
    batch['attention_mask'] = torch.cat((batch['attention_mask'], (labels != -100)), dim=-1)
    if 'position_ids' in batch:
        batch['position_ids'] = batch['attention_mask'].cumsum(-1) - 1
        batch['position_ids'].masked_fill_(batch['attention_mask'] == 0, 1)
    return batch
