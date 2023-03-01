import torch
import os
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer
from mttl.models.encoder_decoder import EncoderDecoder
from mttl.config import Config


if __name__ == '__main__':
    # dummy data dir
    os.environ["NI_DATA_DIR"] = "/tmp/"

    _args = Config("ni/pretrain_short.json+ni/poly_lora.json")
    _args.n_tasks = 768
    _args.model = "t5-small"
    _args.warmup_steps = 0
    _args.learning_rate = 1e-3

    seed_everything(0)
    model = EncoderDecoder(**vars(_args), tokenizer=AutoTokenizer.from_pretrained(_args.model))

    seed_everything(0)
    batch = {
        "input_ids": torch.randint(10, 400, (8, 100)),
        "target_ids": torch.randint(10, 400, (8, 100)),
        "task_ids": torch.randint(0, 768, (8,)).long(),
    }

    optimizers = model.configure_optimizers()
    output = model.teacher_force_step(batch)
    output.backward()

    optimizers["optimizer"].step()

    s = 0
    for p in model.parameters():
        s += p.sum().item()
    assert float(s) / 1e6 == 2.1144700036021518
