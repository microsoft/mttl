#!/bin/bash

MAX_RETRIES=3

run_training() {
  local doc_id=$1
  local gpu=$2
  local config_file=$3
  local output_dir=$4

  local attempt=1
  while [ $attempt -le $MAX_RETRIES ]; do
    echo "Starting attempt $attempt for $doc_id on GPU $gpu"

    mkdir -p "$output_dir/$doc_id"

    CUDA_VISIBLE_DEVICES=$gpu python train_km_simple.py \
      -c "$config_file" \
      -k finetune_task_name="$doc_id" \
      -k wandb_run_name="$AMLT_EXPERIMENT_NAME-$doc_id" \
      -k output_dir="$output_dir/$doc_id"

    if [ $? -eq 0 ]; then
      echo "Training for $doc_id succeeded."
      return 0
    fi

    echo "Training for $doc_id failed (attempt $attempt)."
    attempt=$((attempt + 1))
  done

  echo "Exceeded max retries for $doc_id. Skipping."
  return 1
}

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
NUM_GPUS_PER_NODE=${5:-1}

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

# Function to wait if we already have $NUM_GPUS_PER_NODE background jobs
wait_for_slot() {
  while [ "$(jobs -p | wc -l)" -ge "$NUM_GPUS_PER_NODE" ]; do
    sleep 1
  done
}

GPU_INDEX=0
PIDS=()

for DOC_ID in "$DOCUMENT_IDS[@]"; do
    wait_for_slot

    echo "Starting training for $DOC_ID on GPU $GPU."
    GPU=$((GPU_INDEX % NUM_GPUS_PER_NODE))
    GPU_INDEX=$((GPU_INDEX + 1))

    # Launch training in the background on a specific GPU
    run_training "$DOC_ID" "$GPU" "$CONFIG_FILE" "$OUTPUT_DIR" &
done

# Wait for all background jobs to complete
wait
echo "All training processes finished."

# Now create a library of the best models
python utils/create_library_from_path.py \
  --ckpt_path ${OUTPUT_DIR} \
  --library_path local:///mnt/output/kms/library-${CONFIG_ID}

python utils/create_library_from_path.py \
  --ckpt_path ${OUTPUT_DIR} \
  --library_path az://mttldata/library-${CONFIG_ID}
