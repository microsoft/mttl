import argparse
import json
import os
import time

from projects.kms.deductron.sgl_utils import SGLGeneratorClient
import torch
from transformers import AutoTokenizer

from projects.kms.deductron.launch_sgl import SGLGenerator


def load_args(args_file):
    with open(args_file, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with args.json and (optionally) saved model")
    parser.add_argument("--output-file", type=str, default="generation_output.jsonl", help="File to store outputs")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of validation samples to generate")
    parser.add_argument("--load-mode", type=str, choices=["base", "saved"], default="saved", help="Load base model or saved model")
    args = parser.parse_args()

    # Load training args from file (assumed to be saved via save_args)
    args_path = os.path.join(args.model_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"{args_path} not found")
    train_args = load_args(args_path)
    
    load_mode = args.load_mode
    if load_mode == "saved":
        model_identifier = os.path.join(args.model_dir, "model")
    else:
        model_identifier = train_args.get("m")

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    
   # Start SGL server with the chosen model identifier and seed from training args
    seed = train_args.get("s", 42)
    SGLGenerator(model_identifier, seed)
    generator = SGLGeneratorClient(model_identifier)

    # Setup dataset using the dataset name from training args and the tokenizer
    from projects.kms.deductron.data_utils import prepare_dataset
    from projects.kms.deductron.algos.task import get_task

    block_size = 2048
    _, val_dataset, _ = prepare_dataset(train_args.get("dataset", "nqa"), tokenizer, block_size=block_size)

    # Select a subset from the validation set
    num_samples = min(args.num_samples, len(val_dataset))
    samples = [val_dataset[i] for i in range(num_samples)]
    task = get_task(train_args.get("task"))

    results = []
    for sample in samples:
        messages = task.encode_template([sample["source"]])[0]
        # Use temperature, top_p, and max_tokens from training args if available
        temperature = train_args.get("t", 1.0)
        max_tokens = train_args.get("maxtok", 128)
        top_p = 1.0

        # Query SGL server using its chat method (returns a list of responses)
        output = generator.chat(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        results.append({
            "prompt": messages,
            "response": output[0] if output else ""
        })

    # Save all responses in a JSON-lines file
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"Data generation completed. Results saved to {args.output_file}")

    # Shutdown SGL server
    generator.shutdown()


if __name__ == "__main__":
    main()
