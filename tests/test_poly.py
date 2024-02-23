import os
import torch
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer
from mttl.models.encoder_decoder import EncoderDecoder
from mttl.config import Config
from pathlib import Path


def test_poly():
    root = Path(__file__).parent
    _args = Config(
        f"{root}/configs/t0/pretrain.json+{root}/configs/t0/poly_lora.json",
    )
    _args.n_tasks = 768
    _args.model = "t5-small"
    _args.warmup_steps = 0
    _args.learning_rate = 1e-3

    seed_everything(0)
    model = EncoderDecoder(
        **vars(_args), tokenizer=AutoTokenizer.from_pretrained(_args.model)
    )

    seed_everything(0)
    batch = {
        "input_ids": torch.randint(10, 400, (8, 100)),
        "labels": torch.randint(10, 400, (8, 100)),
        "task_ids": torch.randint(0, 768, (8,)).long(),
        "attention_mask": torch.randint(0, 2, (8, 100)),
    }

    optimizers = model.configure_optimizers()
    output = model.teacher_force_step(batch)
    output.backward()

    optimizers["optimizer"].step()

    s = 0
    for p in model.parameters():
        s += p.sum().item()
    assert round(s / 1e6, 4) == 2.1144
