#!/bin/bash

run_training() {
  local doc_id=$1
  local gpu=$2
  local config_file=$3
  local output_dir=$4
  local lock_dir=$5

  mkdir -p "$output_dir/$doc_id"

  CUDA_VISIBLE_DEVICES=$gpu python train_km_simple.py \
    -c "$config_file" \
    -k finetune_task_name="$doc_id" \
    -k wandb_run_name="$AMLT_EXPERIMENT_NAME-$doc_id" \
    -k output_dir="$output_dir/$doc_id"

  rm -f "${lock_dir}/${doc_id}.lock"

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
LOCK_DIR=/mnt/output/kms/${CONFIG_ID}-locks/

mkdir -p "$LOCK_DIR"

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
ALL_DOCUMENT_IDS=$(awk '{
    print $0
}' "input.txt")

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

ALL_DOCUMENT_ARRAY=($ALL_DOCUMENT_IDS)
ALL_DOCS=${#ALL_DOCUMENT_ARRAY[@]}

GPU_INDEX=0
while :; do
  COMPLETED=$(count_last_models)
  echo "Worker $WORKER_ID sees $COMPLETED / $ALL_DOCS completed."

  if [ "$COMPLETED" -ge "$ALL_DOCS" ]; then
    echo "All docs done."
    break
  fi

  CLAIMED_DOC=""
  for DOC_ID in "${ALL_DOCUMENT_ARRAY[@]}"; do
    if [ -e "$OUTPUT_DIR/$DOC_ID/last_model" ] || [ -e "$LOCK_DIR/$DOC_ID.lock" ]; then
      continue
    fi

    if ( set -o noclobber; echo "$WORKER_ID" > "$LOCK_DIR/$DOC_ID.lock" ) 2>/dev/null; then
      CLAIMED_DOC="$DOC_ID"
      break
    fi
  done

  if [ -z "$CLAIMED_DOC" ]; then
    echo "Worker $WORKER_ID: no doc to claim, waiting..."
    sleep 5
    continue
  fi

  wait_for_slot
  GPU=$((GPU_INDEX % NUM_GPUS_PER_NODE))
  GPU_INDEX=$((GPU_INDEX + 1))

  echo "Worker $WORKER_ID starts training $CLAIMED_DOC on GPU $GPU."
  run_training "$CLAIMED_DOC" "$GPU" "$CONFIG_FILE" "$OUTPUT_DIR" "$LOCK_DIR" &
  sleep 1
done

# Wait for all background jobs to complete
wait

echo "All training processes finished."
sleep 10

# if count of all models is equal to all documents, and this is worker 0, then we can create the library
# else we can just exit
if [ "$WORKER_ID" -ne 0 ]; then
  echo "Worker $WORKER_ID is not worker 0. Exiting."
  exit 0
fi

# Now create a library of the best models
python utils/create_library_from_path.py \
  --ckpt_path ${OUTPUT_DIR} \
  --library_path local:///mnt/output/kms/library-${CONFIG_ID}
