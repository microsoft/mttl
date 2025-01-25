#!/bin/bash

MAX_RETRIES=3

run_eval() {
  local library_uri=$1
  local gpu=$2
  local eval_file=$3
  local output_dir=$4
  local attempt=1

  config_id=$(basename "$eval_file" .json)
  mkdir -p "$output_dir/$config_id"


  while [ $attempt -le $MAX_RETRIES ]; do
    echo "Starting attempt $attempt for $config_id on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python eval_qa.py \
      -c "$eval_file" \
      -k output_dir="$output_dir/$config_id" \
      -k library_id="$library_uri"

    if [ $? -eq 0 ]; then
      echo "Eval for $config_id succeeded."
      return 0
    fi

    echo "Eval for $config_id failed (attempt $attempt)."
    attempt=$((attempt + 1))
  done

  echo "Exceeded max retries for $config_id. Skipping."
  return 1
}


# Check if the correct number of arguments is provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <worker_id> <num_workers> <job_id> <eval_files> [num_gpus_per_node]"
    exit 1
fi

# Assign command-line arguments to variables
WORKER_ID=$1
NUM_WORKERS=$2
JOB_ID=$3
EVAL_FILES=$4
NUM_GPUS_PER_NODE=${5:-1}

echo "WORKER_ID: $WORKER_ID"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "JOB_ID: $JOB_ID"
echo "EVAL_FILES: $EVAL_FILES"
echo "NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"

# Assumes this library is present in the blob storage
LIBRARY_URI=local:///mnt/output/kms/library-${JOB_ID}/
OUTPUT_DIR=/mnt/output/kms/evals/library-${JOB_ID}/

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

export PYTHONPATH=$PWD/../../
ls -l $PWD/../../

# Function to wait if we already have $NUM_GPUS_PER_NODE background jobs
wait_for_slot() {
  while [ "$(jobs -p | wc -l)" -ge "$NUM_GPUS_PER_NODE" ]; do
    sleep 1
  done
}

# comma separated
IFS=',' read -r -a EVAL_ARRAY <<< "$EVAL_FILES"

GPU_INDEX=0
PIDS=()

for EVAL_FILE in "${EVAL_ARRAY[@]}"; do
    wait_for_slot

    GPU=$((GPU_INDEX % NUM_GPUS_PER_NODE))
    GPU_INDEX=$((GPU_INDEX + 1))

    echo "Starting evaluating $LIBRARY_URI on GPU $GPU."

    # Launch training in the background on a specific GPU
    run_eval "$LIBRARY_URI" "$GPU" "$EVAL_FILE" "$OUTPUT_DIR" &
done

# Wait for all background jobs to complete
wait
echo "All eval processes finished."
