import argparse
import os

from datasets import concatenate_datasets, load_from_disk


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate arrow datasets in a directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing arrow datasets",
    )
    parser.add_argument("--hf_id", type=str, required=True, help="HF dataset ID")

    args = parser.parse_args()

    splits_datasets = {}

    # Loop over all subdirectories in the input directory
    for item in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path):
            try:
                dataset = load_from_disk(item_path)
                for split in dataset.keys():
                    if split not in splits_datasets:
                        splits_datasets[split] = []
                    splits_datasets[split].append(dataset[split])
                print(f"Loaded dataset from {item_path}")
            except Exception as e:
                print(f"Failed to load dataset from {item_path}: {e}")
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
    concatenated_dataset.push_to_hub(args.hf_id)
    print(f"Pushing concatenated dataset to {args.hf_id}")


if __name__ == "__main__":
    main()
