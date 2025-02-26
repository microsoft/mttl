#!/bin/bash
MAX_RETRIES=3

run_task() {
  local task_cmd=$1
  local attempt=1

  while [ $attempt -le $MAX_RETRIES ]; do
    echo "Attempt $attempt: Running $task_cmd"
    # Run the command with the assigned GPU via CUDA_VISIBLE_DEVICES.
    bash -c "$task_cmd"
    if [ $? -eq 0 ]; then
      echo "Task succeeded on GPU $gpu: $task_cmd"
      return 0
    fi
    echo "Task failed (attempt $attempt) on GPU $gpu: $task_cmd"
    attempt=$((attempt + 1))
  done

  echo "Exceeded max retries for task on GPU $gpu: $task_cmd"
  return 1
}

if [ $# -lt 1 ]; then
  echo "Usage: $0 <TASK_FILE>"
  exit 1
fi

TASK_FILE=$1

if [ ! -f "$TASK_FILE" ]; then
  echo "Error: TASK_FILE '$TASK_FILE' not found."
  exit 1
fi

while IFS= read -r task_cmd || [ -n "$task_cmd" ]; do
  # Skip empty lines and comments
  [[ -z "$task_cmd" || "$task_cmd" =~ ^# ]] && continue

  echo "Starting task: $task_cmd"
  run_task "$task_cmd"
done < "$TASK_FILE"

wait
echo "All tasks finished."