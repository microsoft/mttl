import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)
args = parser.parse_args()

# Load the JSON data from a file
with open(args.file, "r") as f:
    data = json.load(f)

# Define the keys in the desired order
key_order = [
    "piqa",
    "boolq",
    "winogrande",
    "hellaswag",
    "arc-easy",
    "arc-challenge",
    "humaneval",
    "openbookqa",
    "bbh-fast",
    "mbpp",
    "mean",
]

# Extract scores, format to three decimal places, and join with &
scores = []
for key in key_order:
    score = data.get(key)
    if score is not None:
        scores.append(f"{score * 100:.1f}")
    else:
        scores.append("N/A")

# Print all scores on one line separated by &
print(" & ".join(scores))
