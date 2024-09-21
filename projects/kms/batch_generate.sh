#!/bin/bash

WORKER_IDX=$1
NUM_WORKERS=$2
NUM_DOCUMENTS=$3

DOCS_PER_WORKER=$(((NUM_DOCUMENTS+NUM_WORKERS-1)/NUM_WORKERS))
FIRST_DOC=$((WORKER_IDX*DOCS_PER_WORKER+1))
LAST_DOC=$((FIRST_DOC+DOCS_PER_WORKER))
LAST_DOC=$((LAST_DOC>NUM_DOCUMENTS ? NUM_DOCUMENTS : LAST_DOC))
NUM_DOCS=$((LAST_DOC-FIRST_DOC+1))

i=0
pass=0
while read DOCUMENT_ID; do
    ((i++))
    if [[ $i -lt $FIRST_DOC || $i -gt $LAST_DOC ]]; then
        continue
    fi
    echo "Running for document $DOCUMENT_ID"
    python generate_for_dataset.py --dataset_type narrativeqa --dataset_task $DOCUMENT_ID --use_prompts summary,qa --output_path $AMLT_OUTPUT_DIR/$DOCUMENT_ID
    if [[ $? == 0 ]]; then
        ((pass++))
    fi
done < nqa_ids.txt

echo "$pass / $NUM_DOCS completed successfully"
if [[ $pass != $NUM_DOCS ]]; then
    exit 1
fi