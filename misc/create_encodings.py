import pickle
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from mttl.config import parse_config
from mttl.datamodule.ni_data_module import NIDataModule
from mttl.datamodule.t0_data_module import T0PretrainDataModule, T0FinetuneDataModule
from mttl.models.encoder_decoder import EncoderDecoder
from mttl.models.t0_encoder_decoder import T0EncoderDecoder
from mttl.utils import trim_batch, get_checkpoint_path


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


def convert_dataset(
    dataset, tokenizer, model, start=0, stop=-1
):
    batch = []
    hash = []
    encodings = []
    task_ids = []

    if stop == -1:
        stop = len(dataset)
    else:
        stop = min(len(dataset), stop)

    sentence_encoding = isinstance(model, SentenceTransformer)
    encode_func = sentence_encode_batch if sentence_encoding else encode_batch

    for example_num in tqdm(range(start, stop)):
        example_info = dataset[example_num]
        batch.append(
            example_info.input_text if sentence_encoding else example_info.input_ids
        )
        hash.append(example_info.hash)
        task_ids.append(example_info.task_id)

        if len(batch) == 128:
            encodings.extend(encode_func(batch, model, tokenizer))
            batch = []

    if len(batch):
        encodings.extend(encode_func(batch, model, tokenizer))
        batch = []
    return hash, encodings, task_ids


if __name__ == "__main__":
    config = parse_config()
    sentence_encoder = False

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
            model = AutoModelForSeq2SeqLM.from_pretrained(config.model, cache_dir="/tmp/cache").cuda()
            os.system("/bin/rm -rf /tmp/cache")  # free-up space

    # instantiate T0 data
    if config.dataset == "ni":
        config.custom_tasks_splits = "./dataloader/ni_data/train_tasks.txt"
        dm = NIDataModule(config)
        id2task = dict((k, v) for v, k in dm.task2id.items())
        dm.setup("fit")

        tr_data = [[], [], [], [], []]
        for dataset in [dm.train_dataset, dm.val_dataset, dm.test_dataset]:
            h, e, t = convert_dataset(dataset, dm.tokenizer, model)
            tr_data[0].extend(h)
            tr_data[1].extend(e)
            tr_data[2].extend(t)
            tr_data[3].extend([id2task[tid] for tid in t])
            tr_data[4].extend(0 for _ in range(len(h)))

        # for test tasks
        config.custom_tasks_splits = "./dataloader/ni_data/test_tasks.txt"
        dm = NIDataModule(config)
        dm.setup("fit")

        for dataset in [dm.train_dataset, dm.val_dataset, dm.test_dataset]:
            h, e, t = convert_dataset(dataset, dm.tokenizer, model)
            tr_data[0].extend(h)
            tr_data[1].extend(e)
            tr_data[2].extend(t)
            tr_data[3].extend([dm.id2task[tid] for tid in t])
            tr_data[4].extend(1 for _ in range(len(h)))  # test data

        save_path = os.path.join(config.output_dir, f"encodings.pkl")
        with open(save_path, "wb") as f:
            tr_data[1] = np.array(tr_data[1]).astype(np.float32)
            pickle.dump(tr_data, f)

    elif config.dataset == "t0":
        dm = T0PretrainDataModule(config)
        dm.setup("fit")

        print("Length of training dataset: ", len(dm.full_train_dataset))
        chunk = 0
        num_processed = 0
        next_task = 0

        
        while num_processed < len(dm.full_train_dataset):
            tr_data = [[], [], [], [], []]

            save_path = os.path.join(
                config.output_dir,
                f"encodings.pkl-chunk{chunk}",
            )
            h, e, t = convert_dataset(
                dm.full_train_dataset,
                dm.tokenizer,
                model,
                start=(chunk - 1) * 256_000,
                stop=chunk * 256_000,
            )
            tr_data[0].extend(h)
            tr_data[1].extend(e)
            tr_data[2].extend(t)
            tr_data[3].extend([dm.id2task[tid] for tid in t])
            tr_data[4].extend(0 for _ in range(len(h)))

            with open(save_path, "wb") as f:
                tr_data[1] = np.array(tr_data[1]).astype(np.float32)
                pickle.dump(tr_data, f)

            chunk += 1
            next_task = np.max(tr_data[2] + [next_task])
            num_processed += len(tr_data[0])
            tr_data = [[], [], [], [], []]

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

            dataset = dm.flattened_datasets()

            h, e, _ = convert_dataset(dataset, dm.tokenizer, model)
            tr_data[0].extend(h)
            tr_data[1].extend(e)
            tr_data[2].extend([next_task for _ in range(len(h))])
            tr_data[3].extend([finetune_task_name for _ in h])
            tr_data[4].extend(1 for _ in range(len(h)))  # test data

        # save last chunk
        save_path = os.path.join(
            config.output_dir,
            f"encodings.pkl-chunk{chunk}",
        )
        with open(save_path, "wb") as f:
            tr_data[1] = np.array(tr_data[1]).astype(np.float32)
            pickle.dump(tr_data, f)
