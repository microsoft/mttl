#!/bin/bash

# Assign command-line arguments to variables
WORKER_ID=$1
NUM_WORKERS=$2
MODEL_TYPE=$3
OUTPUT_PATH=$4
DATASET_TYPE=$5

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
jq -r '.[] | .[]' "splits/${DATASET_TYPE}/${DATASET_TYPE}_full.json" > input.txt

# Extract IDs assigned to this worker
DOCUMENT_IDS=$(awk -v wid=$WORKER_ID -v nworkers=$NUM_WORKERS '{
    if ((NR - 1) % nworkers == wid) print $0
}' "input.txt" | paste -sd, -)

echo $DOCUMENT_IDS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python \
    generate_for_dataset.py \
    -k model=$MODEL_TYPE \
    -k model_type=local \
    -k dataset_type=$DATASET_TYPE \
    -k dataset_task=$DOCUMENT_IDS \
    -k use_prompts=summary,qa \
    -k output_path=$OUTPUT_PATH/$WORKER_ID
