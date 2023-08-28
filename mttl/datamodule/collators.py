from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from typing import List, Union, Optional
import torch

from mttl.dataloader.data_utils import ExampleInfo
from mttl.datamodule.utils import prepare_inputs_for_gpt_family


@dataclass
class DefaultCollator():  
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_input_length: Optional[int] = None
    max_output_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    model_family: str = "seq2seq"

    def __call__(self, batch: List[ExampleInfo]):
        inputs = [b.input for b in batch]
        targets = [b.target for b in batch]
        # Add space for auto-regressive model tokenization
        targets = [' ' + l for l in targets]
        # Remove multiple spaces, which mess with tiktoken (?)
        inputs = [' '.join(s.split()) for s in inputs]
        hashes = [b.hash for b in batch]
        task_ids = [b.task_id for b in batch]
        instruction_hashes = [b.instruction_hash for b in batch]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        # model_inputs["inputs"] = inputs
        # model_inputs["labels_text"] = targets
        targets = self.tokenizer(
            targets,
            max_length=self.max_output_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
        )
        label_mask = targets["attention_mask"].bool()
        model_inputs["labels"] = targets["input_ids"].masked_fill(
            ~label_mask, self.label_pad_token_id
        )
        model_inputs["hashes"] = hashes
        model_inputs["instruction_hashes"] = instruction_hashes
        model_inputs["task_ids"] = torch.LongTensor(task_ids)

        if self.model_family == "gpt":
            model_inputs = prepare_inputs_for_gpt_family(model_inputs, self.tokenizer)
        return model_inputs
