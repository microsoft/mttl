import argparse
import os
from collections import defaultdict

from datasets import concatenate_datasets, load_dataset, load_from_disk


def convert_str_to_python(python_str):
    try:
        # execute the python string
        out = eval(python_str)
        assert isinstance(out, list), "The python string should evaluate to a list."
        assert len(out) > 1, "The list should have at least two elements."
    except Exception as e:
        print(f"Failed to convert string to python object: {e}")
        out = None

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate arrow datasets in a directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing arrow datasets",
    )
    parser.add_argument(
        "--input_hf_ids",
        type=str,
        nargs="+",
        help="HF dataset IDs to concatenate",
    )
    parser.add_argument("--hf_id", type=str, required=True, help="HF dataset ID")
    args = parser.parse_args()

    # make sure only one of `input_hf_ids` or `input_idr` is provided
    if args.input_dir and args.input_hf_ids:
        raise ValueError(
            "Only one of `input_dir` or `input_hf_ids` should be provided."
        )

    if not args.input_dir and not args.input_hf_ids:
        raise ValueError("One of `input_dir` or `input_hf_ids` should be provided.")

    if args.input_hf_ids:
        if len(args.input_hf_ids) == 1:
            args.input_hf_ids = convert_str_to_python(args.input_hf_ids[0])

    splits_datasets = defaultdict(list)

    if args.input_dir:
        # Loop over all subdirectories in the input directory
        for item in os.listdir(args.input_dir):
            item_path = os.path.join(args.input_dir, item)
            if os.path.isdir(item_path):
                try:
                    dataset = load_from_disk(item_path)
                    for split in dataset.keys():
                        splits_datasets[split].append(dataset[split])
                    print(f"Loaded dataset from {item_path}")
                except Exception as e:
                    print(f"Failed to load dataset from {item_path}: {e}")
                    continue
    elif args.input_hf_ids:
        # I will not explicitly check for the splits
        for hf_id in args.input_hf_ids:
            try:
                dataset = load_dataset(hf_id)
                for split in dataset.keys():
                    splits_datasets[split].append(dataset[split])
            except Exception as e:
                print(f"Failed to load dataset from {hf_id}: {e}")
                continue

    if not splits_datasets:
        print("No datasets found to concatenate.")
        return

    # Concatenate datasets for each split
    concatenated_splits = {}
    for split, datasets_list in splits_datasets.items():
        concatenated_dataset = concatenate_datasets(datasets_list)
        concatenated_splits[split] = concatenated_dataset
        print(f"Concatenated {split} split with {len(datasets_list)} datasets")
        print(f"Length of concatenated dataset: {len(concatenated_splits[split])}")

    # Save concatenated dataset
    breakpoint()
    concatenated_dataset.push_to_hub(args.hf_id)
    print(f"Pushing concatenated dataset to {args.hf_id}")


if __name__ == "__main__":
    main()
