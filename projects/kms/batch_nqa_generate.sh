#!/bin/bash

# Assign command-line arguments to variables
WORKER_ID=$1
NUM_WORKERS=$2

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
}' "nqa_ids.txt" | paste -sd, -)


echo $DOCUMENT_IDS
CUDA_VISIBLE_DEVICES=0 python generate_for_dataset.py --dataset sordonia/narrativeqa_sanitized --model microsoft/Phi-3-mini-4k-instruct --dataset_type narrativeqa --dataset_task $DOCUMENT_IDS --use_prompts summary,qa --output_path $AMLT_OUTPUT_DIR
