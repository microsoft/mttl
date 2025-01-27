#!/bin/bash

run_training() {
  local doc_id=$1
  local gpu=$2
  local config_file=$3
  local output_dir=$4

  mkdir -p "$output_dir/$doc_id"

  CUDA_VISIBLE_DEVICES=$gpu python train_km_simple.py \
    -c "$config_file" \
    -k finetune_task_name="$doc_id" \
    -k wandb_run_name="$AMLT_EXPERIMENT_NAME-$doc_id" \
    -k output_dir="$output_dir/$doc_id"

  if [ -e "$output_dir/$doc_id/last_model" ]; then
    echo "Training for $doc_id succeeded."
    return 0
  fi

  return 1
}

# Check if the correct number of arguments is provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <worker_id> <num_workers> <task_file> <config_file>"
    exit 1
fi

# Assign command-line arguments to variables
WORKER_ID=$1
NUM_WORKERS=$2
TASK_FILE=$3
CONFIG_FILE=$4
NUM_GPUS_PER_NODE=${5:-1}

CONFIG_FILE=$CONFIG_FILE
CONFIG_ID=$(basename $CONFIG_FILE .json)
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
jq -r '.[] | .[]' "$TASK_FILE" > input.txt

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

# Function to understand if we have already trained the model
count_last_models() {
  find "$OUTPUT_DIR" -type d -name "last_model" | wc -l
}

DOC_ARRAY=($DOCUMENT_IDS)
TOTAL_DOCS=${#DOC_ARRAY[@]}

GPU_INDEX=0
while :; do
  COMPLETED=$(count_last_models)
  echo "Completed models: $COMPLETED / $TOTAL_DOCS"

  if [ "$COMPLETED" -ge "$TOTAL_DOCS" ]; then
    echo "All documents have last_model. Done."
    break
  fi

  for DOC_ID in $DOCUMENT_IDS; do
      if [ ! -e "$OUTPUT_DIR/$DOC_ID/last_model" ]; then
        wait_for_slot

        GPU=$((GPU_INDEX % NUM_GPUS_PER_NODE))
        GPU_INDEX=$((GPU_INDEX + 1))

        echo "Starting training for $DOC_ID on GPU $GPU."

        # Launch training in the background on a specific GPU
        run_training "$DOC_ID" "$GPU" "$CONFIG_FILE" "$OUTPUT_DIR" &
        sleep 1
      else
        echo "Model for $DOC_ID already exists. Skipping."
      fi
  done

  # Wait for all background jobs to complete
  wait
done


echo "All training processes finished."

# Now create a library of the best models
python utils/create_library_from_path.py \
  --ckpt_path ${OUTPUT_DIR} \
  --library_path local:///mnt/output/kms/library-${CONFIG_ID}
