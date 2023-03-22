import functools
import pickle
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from torch.utils.data import DataLoader
from mttl.config import parse_config
from mttl.cluster_tuning.encodings import Encodings
from mttl.datamodule.ni_data_module import NIDataModule
from mttl.datamodule.t0_data_module import T0PretrainDataModule, T0FinetuneDataModule
from mttl.models.encoder_decoder import EncoderDecoder
from mttl.models.t0_encoder_decoder import T0EncoderDecoder
from mttl.utils import trim_batch, get_checkpoint_path, hash_example


@torch.no_grad()
def sentence_encode_batch(batch, model, _):
    out = model.encode(batch, convert_to_numpy=True)
    return out.tolist()


@torch.no_grad()
def encode_batch(batch, model, tokenizer):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    input_ids = trim_batch(input_ids, tokenizer.pad_token_id).cuda()
    attention_mask = (input_ids != tokenizer.pad_token_id).float()  # [bs, max_seq_len]
    encoder_hidden_states = model.encoder(
        input_ids=input_ids, attention_mask=attention_mask
    )
    out = encoder_hidden_states[0]  # (bs, sq, D)
    out = (out * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
        -1, keepdim=True
    )
    return out.cpu().numpy().tolist()


def collate(batch):
    input_ids = [b.input_ids for b in batch]
    input_text = [b.input_text for b in batch]
    hashes = [b.hash for b in batch]
    task_ids = [b.task_id for b in batch]

    return (input_ids[0], input_text[0], hashes[0], task_ids[0])


def convert_dataset(
    datamodule,
    tokenizer,
    model,
    batch_size=-1,
    use_description=False,
):
    sentence_encoding = isinstance(model, SentenceTransformer)
    encode_func = sentence_encode_batch if sentence_encoding else encode_batch

    task_ids = []
    batch = []
    hash = []
    encodings = []

    if use_description:
        inputs = datamodule.all_instructions

        # just encode it all at once (smaller number of them usually)
        for input in inputs:
            example_hash = hash_example(input)
            if sentence_encoding:
                example_input = input
            else:
                # need to tokenize the instruction
                example_input = tokenizer(input, return_tensors="pt").input_ids.squeeze(0)

            task_id = 0
            batch.append(example_input)
            hash.append(example_hash)
            task_ids.append(task_id)

        encodings = encode_func(batch, model, tokenizer)
        yield hash, encodings, task_ids
    else:
        dataset = datamodule.full_dataset

        # create an ad-hoc loader for the dataset
        loader = DataLoader(
            dataset,
            num_workers=16,
            batch_size=1,
            collate_fn=collate,
            pin_memory=True,
            shuffle=False,
        )

        task_ids = []
        batch = []
        hash = []

        for example_info in loader:
            if sentence_encoding:
                example_input = example_info[1]
                example_hash = example_info[2]
            else:
                example_input = example_info[0]
                example_hash = example_info[2]
            example_task_id = example_info[-1]

            batch.append(example_input)
            hash.append(example_hash)
            task_ids.append(example_task_id)

            if len(batch) == batch_size:
                print("Encoding batch:", hash[-1])
                encodings = encode_func(batch, model, tokenizer)
                yield hash, encodings, task_ids
                batch, hash, task_ids = [], [], []

        if len(batch):
            encodings = encode_func(batch, model, tokenizer)
            yield hash, encodings, task_ids
            batch, hash, task_ids = [], [], []


def encode_ni(config, model):
    config.custom_tasks_splits = "./dataloader/ni_data/train_tasks.txt"

    dm = NIDataModule(config)
    dm.setup("fit")

    data = Encodings(input_type="instruction" if config.encode_instruction else "input")
    for h, e, t in convert_dataset(
        dm, dm.tokenizer, model, batch_size=512, use_description=config.encode_instruction
    ):
        data.hashes.extend(h)
        data.encodings.extend(e)
        data.task_ids.extend(t)
        data.task_names.extend([dm.id2task[tid] for tid in t])
        data.is_test.extend(0 for _ in range(len(h)))

    # for test tasks
    config.custom_tasks_splits = "./dataloader/ni_data/test_tasks.txt"
    dm = NIDataModule(config)
    dm.setup("fit")

    for h, e, t in convert_dataset(
        dm, dm.tokenizer, model, batch_size=512, use_description=config.encode_instruction
    ):
        data.hashes.extend(h)
        data.encodings.extend(e)
        data.task_ids.extend(t)
        data.task_names.extend([dm.id2task[tid] for tid in t])
        data.is_test.extend(1 for _ in range(len(h)))

    data.save(os.path.join(config.output_dir, f"encodings.pkl"))


def encode_t0(config, model):
    dm = T0PretrainDataModule(config)
    dm.setup("fit")

    print("Length of multi-task training set: ", len(dm.full_dataset))

    data = Encodings(input_type="instruction" if config.encode_instruction else "input")
    next_task = 0
    chunk = 0

    for h, e, t in convert_dataset(
        dm, dm.tokenizer, model, batch_size=256, use_description=config.encode_instruction
    ):
        data.hashes.extend(h)
        data.encodings.extend(e)
        data.task_ids.extend(t)
        data.task_names.extend([dm.id2task[tid] for tid in t])
        data.is_test.extend(0 for _ in range(len(h)))

        if len(data.encodings) == 256_000:
            save_path = os.path.join(
                config.output_dir,
                f"encodings.pkl-chunk{chunk}",
            )
            next_task = np.max(data.task_ids + [next_task])
            chunk += 1
            data.save(save_path)
            data.clear()

    if len(data.encodings):
        save_path = os.path.join(
            config.output_dir,
            f"encodings.pkl-chunk{chunk}",
        )
        next_task = np.max(data.task_ids + [next_task])
        chunk += 1
        data.save(save_path)
        data.clear()

    for finetune_task_name in [
        "copa",
        "h-swag",
        "storycloze",
        "winogrande",
        "wsc",
        "wic",
        "rte",
        "cb",
        "anli-r1",
        "anli-r2",
        "anli-r3",
    ]:
        next_task += 1
        config.finetune_task_name = finetune_task_name

        dm = T0FinetuneDataModule(config)
        dm.setup("fit")

        for h, e, _ in convert_dataset(
            dm,
            dm.tokenizer,
            model,
            batch_size=256,
            use_description=config.encode_instruction,
        ):  
            data.hashes.extend(h)
            data.encodings.extend(e)
            data.task_ids.extend([next_task for _ in range(len(h))])
            data.task_names.extend([finetune_task_name for _ in h])
            data.is_test.extend(1 for _ in range(len(h)))

    # save last chunk
    save_path = os.path.join(
        config.output_dir,
        f"encodings.pkl-chunk{chunk}",
    )
    data.save(save_path)


if __name__ == "__main__":
    config = parse_config(raise_error=False)

    if config.checkpoint:
        checkpoint = get_checkpoint_path(config.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(config.model)

        if config.dataset == "ni":
            model = EncoderDecoder.load_from_checkpoint(
                checkpoint, tokenizer=tokenizer
            ).model.cuda()
        elif config.dataset == "t0":
            model = T0EncoderDecoder.load_from_checkpoint(
                checkpoint, tokenizer=tokenizer
            ).model.cuda()
    else:
        if config.model == "all-mpnet-base-v2":
            model = SentenceTransformer(config.model, device="cuda")
            config.model = "t5-small"  # to avoid errors downstream for tokenization
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.model, cache_dir=os.environ.get("TRANSFORMERS_CACHE", "/tmp/cache")
            ).cuda()
            os.system("/bin/rm -rf /tmp/cache")  # free-up space

    if config.dataset == "ni":
        encode_ni(config, model)

    elif config.dataset == "t0":
        encode_t0(config, model)
