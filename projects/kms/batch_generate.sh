#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <worker_id> <num_workers> <input_file> <dataset_dir>"
    exit 1
fi

# Assign command-line arguments to variables
WORKER_ID=$1
NUM_WORKERS=$2
INPUT_FILE=$3
DATASET_DIR=$4

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

# Extract IDs assigned to this worker
DOCUMENT_IDS=$(awk -v wid=$WORKER_ID -v nworkers=$NUM_WORKERS '{
    if ((NR - 1) % nworkers == wid) print $0
}' "$INPUT_FILE" | paste -sd, -)

CUDA_VISIBLE_DEVICES=0 python generate_for_dataset.py \
    --dataset $DATASET_DIR \
    --dataset_type narrativeqa \
    --dataset_task $DOCUMENT_IDS \
    --use_prompts summary,qa \
    --output_path $AMLT_OUTPUT_DIR
