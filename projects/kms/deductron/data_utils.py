from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

from projects.kms.deductron.ddp_utils import ddp_state


class MultiTensorDataset(Dataset):
    """
    A custom Dataset that handles multiple tensors and returns a tuple of indexed elements.

    Args:
        *tensors (torch.Tensor): Variable number of tensors, each of shape (N, ...), where N is the number of samples.
    """

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("At least one tensor must be provided.")

        self.tensors = tensors
        self.length = self.tensors[0].size(0)

        # Ensure all tensors have the same length
        for tensor in self.tensors:
            if tensor.size(0) != self.length:
                raise ValueError(
                    "All tensors must have the same number of samples (dimension 0)."
                )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)


def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of multiple tensors.

    Args:
        batch (list of tuples): Each tuple contains (a[idx], b[idx], c[idx]).

    Returns:
        tuple of tensors: Batched tensors for a, b, and c.
    """
    tensors = zip(*batch)
    stacked_tensors = [torch.stack(t, dim=0) for t in tensors]
    return stacked_tensors


def get_dataloader(
    dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False
):
    """
    Creates a DataLoader with DistributedSampler if in DDP mode.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
        pin_memory (bool, optional): Whether to pin memory. Defaults to True.

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    if ddp_state.ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = (
            torch.utils.data.RandomSampler(dataset)
            if shuffle
            else torch.utils.data.SequentialSampler(dataset)
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling is handled by the sampler
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory and ddp_state.device.startswith("cuda"),
        drop_last=True if ddp_state.ddp else False,
    )
    return dataloader


def chunk_text(
    text: str,
    tokenizer: AutoTokenizer,
    block_size: int = 2048,
    chunk_overlap: Union[float, int] = 0.1,
):
    if isinstance(chunk_overlap, float):
        assert 0.0 <= chunk_overlap < 1.0
        chunk_overlap = int(block_size * chunk_overlap)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=block_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        yield chunk


def get_ending_tokens(tokenizer):
    dummy_template = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "dummy content"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return dummy_template[
        dummy_template.rindex("dummy content") + len("dummy content") :
    ]


def pad_query_and_response(
    queries: list[torch.Tensor],
    responses: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
) -> torch.Tensor:
    seq_len = max([q.shape[0] + r.shape[0] for q, r in zip(queries, responses)])

    # Create an output tensor filled with the padding value
    query_response = torch.full(
        (len(queries), seq_len),
        padding_value,
        dtype=queries[0].dtype,
        device=queries[0].device,
    )
    query_response_mask = torch.full(
        (len(queries), seq_len),
        0,
        dtype=queries[0].dtype,
        device=queries[0].device,
    )
    response_mask = torch.full(
        (len(queries), seq_len),
        0,
        dtype=queries[0].dtype,
        device=queries[0].device,
    )

    for i, (q, r) in enumerate(zip(queries, responses)):
        # Determine the slice for the sequence dimension
        t = torch.cat((q, r), dim=0)
        if padding_side == "left":
            seq_slice = slice(seq_len - t.shape[0], seq_len)
            response_slice = slice(seq_len - r.shape[0], seq_len)
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
            response_slice = slice(q.shape[0], t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,)
        query_response[i][slices] = t
        query_response_mask[i][slices] = 1
        response_mask[i][response_slice] = 1
    return query_response, query_response_mask, response_mask


def create_joint_tensors(
    tokenizer,
    queries: Union[List[Dict[str, str]]],
    responses: List[str],
    is_final=None,
    max_length=4096,
):
    """
    For VPPO, messages can contain also a partial assistant message (the prefix so far),
    we need to be careful in handling this.
    """
    if len(queries) != len(responses):
        raise ValueError("Queries and responses must be the same length.")

    is_final = is_final or [True] * len(queries)

    # if the latest message is not from the assistant, we need to add the assistant token
    ending_tokens = get_ending_tokens(tokenizer)
    tokenized_queries = []

    for query in queries:
        if query[-1]["role"] != "assistant":
            tok_query = torch.tensor(
                tokenizer.apply_chat_template(
                    query,
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
        else:
            tok_query = torch.tensor(
                tokenizer.apply_chat_template(
                    query,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=True,
                )
            )
        tokenized_queries.append(tok_query)

    # response is stripped by apply_chat_template...
    responses = [
        r.strip() + (ending_tokens if f else "") for r, f in zip(responses, is_final)
    ]
    tokenized_responses = [torch.tensor(t) for t in tokenizer(responses)["input_ids"]]
    query_and_response_tensors, query_and_response_mask, response_mask = (
        pad_query_and_response(
            tokenized_queries,
            tokenized_responses,
            tokenizer.pad_token_id,
            padding_side="right",
        )
    )
    if query_and_response_tensors.shape[1] > max_length:
        query_and_response_tensors = query_and_response_tensors[:, :max_length]
        query_and_response_mask = query_and_response_mask[:, :max_length]
        response_mask = response_mask[:, :max_length]
    return query_and_response_tensors, query_and_response_mask, response_mask


class AccumulatorDict:
    """Accumulate values in a dictionary."""

    def __init__(self):
        self.data = {}

    def accumulate(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def mean(self, key):
        return sum(self.data[key]) / len(self.data[key])

    def get(self):
        # return mean values and clear the stats
        data = {k: sum(v) / len(v) for k, v in self.data.items()}
        self.data = {}
        return data

    def clear(self):
        self.data = {}


def prepare_nqa_dataset(tokenizer, block_size=2048):
    from datasets import load_dataset

    dataset = load_dataset("sordonia/narrativeqa_sanitized", split="train")

    def chunk_row(example):
        sources, labels, dids = [], [], []
        for text, did in zip(example["text"], example["document_id"]):
            chunks = list(chunk_text(text, tokenizer, block_size=block_size))
            for i in range(len(chunks) - 1):
                sources.append(chunks[i])
                labels.append(chunks[i + 1])
                dids.append(did)
        return {"source": sources, "label": labels, "document_id": dids}

    new_dataset = dataset.map(
        chunk_row,
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=100,
        num_proc=16,
    )
    return new_dataset
