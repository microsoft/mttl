#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <worker_id> <num_workers> <json_file> <config_id>"
    exit 1
fi

# Assign command-line arguments to variables
WORKER_ID=$1
NUM_WORKERS=$2
JSON_FILE=$3
CONFIG_ID=$4
export WANDB_PROJECT=knowledge-modules-${CONFIG_ID}
export WANDB_MODE="online"

CONFIG_FILE=configs/${CONFIG_ID}.json
OUTPUT_DIR=/mnt/output/kms/${CONFIG_ID}/

# Validate that WORKER_ID and NUM_WORKERS are integers
if ! [[ $WORKER_ID =~ ^[0-9]+$ ]] ; then
   echo "Error: Worker ID must be a non-negative integer"
   exit 1
fi

if ! [[ $NUM_WORKERS =~ ^[0-9]+$ ]] ; then
   echo "Error: Number of workers must be a positive integer"
   exit 1
fi

# Ensure WORKER_ID is less than NUM_WORKERS
if [ $WORKER_ID -ge $NUM_WORKERS ]; then
    echo "Error: Worker ID must be less than the number of workers"
    exit 1
fi

# Flatten the input json file
jq -r '.[] | .[]' "$JSON_FILE" > input.txt

# Extract IDs assigned to this worker
DOCUMENT_IDS=$(awk -v wid=$WORKER_ID -v nworkers=$NUM_WORKERS '{
    if ((NR - 1) % nworkers == wid) print $0
}' "input.txt")

# Check if DOCUMENT_IDS is empty
if [ -z "$DOCUMENT_IDS" ]; then
    echo "No documents assigned to worker $WORKER_ID. Exiting."
    exit 0
fi

export PYTHONPATH=$PWD/../../
ls -l $PWD/../../

IFS=$'\n'
for DOC_ID in $DOCUMENT_IDS; do
    if [ -d "$OUTPUT_DIR/$DOC_ID/best_model" ]; then
        echo "Skipping training."
        continue
    fi

    mkdir -p "$OUTPUT_DIR/$DOC_ID"
    torchrun --nproc-per-node 1 --master_port=$((29500 + $CUDA_VISIBLE_DEVICES)) train_km.py \
        -c "$CONFIG_FILE" \
        -k finetune_task_name="$DOC_ID" \
        -k wandb_run_name="$AMLT_EXPERIMENT_NAME-$DOC_ID" \
        -k output_dir="$OUTPUT_DIR/$DOC_ID"
done
