import hydra
from omegaconf import OmegaConf
from lm import LM
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.dataloader.ni_metrics import rougeL_score
import tqdm
import numpy as np
from alpaca_config import AlpacaConfig


def decode_strip(tokenizer, tensor):
    outputs = []
    for output in tensor:
        outputs.append(tokenizer.decode(
            [
                t
                for t in output
                if t
                not in [
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                ]
            ],
            skip_special_tokens=True,
        ))
    return outputs


def run(args):
    if args.checkpoint is not None:
        checkpoint = LM

    dataset = AlpacaDataModule(args, generation_mode=True)
    dataset.setup()

    if args.checkpoint is not None:
        checkpoint = LM.load_from_checkpoint(
            args.checkpoint, tokenizer=dataset.tokenizer
        ).cuda()
    else:
        checkpoint = LM(**vars(args), tokenizer=dataset.tokenizer).cuda()

    rouges = []
    pbar = tqdm.tqdm(
        total=len(dataset.val_dataloader()),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        desc="Generation",
    )
    for sample in dataset.val_dataloader():
        outputs = checkpoint.model.generate(
            sample["input_ids"].cuda(),
            do_sample=True,
            temperature=1e-5,
            max_new_tokens=args.max_length,
            top_p=0.7,
            return_dict_in_generate=True,
            pad_token_id=dataset.tokenizer.pad_token_id,
            eos_token_id=dataset.tokenizer.eos_token_id,
        )
        outputs = outputs[0][:, sample["input_ids"].shape[1] :]
        target_ids = sample["target_ids"]
        target_ids[target_ids == -100] = dataset.tokenizer.pad_token_id
        inputs = decode_strip(dataset.tokenizer, sample["input_ids"])
        outputs = decode_strip(dataset.tokenizer, outputs)
        targets = decode_strip(dataset.tokenizer, target_ids)
        rouge = rougeL_score(outputs, targets)
        rouges.append(rouge)
        print("Instruction:\n", inputs[0], "\n###########")
        print("Ground Truth:\n", targets[0], "\n###########")
        print("Generation:\n", outputs[0], "\n###########")
        pbar.update(1)
        pbar.set_postfix_str(f"{np.mean(rouges):.1%}")


if __name__ == "__main__":
    args = AlpacaConfig.parse(raise_error=False)
    run(args)
