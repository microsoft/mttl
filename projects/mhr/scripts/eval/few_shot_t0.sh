# NOTE : must assign a value to `PATH_TO_CHECKPOINT` and `PATH_TO_OUTPUT_DIR`
# using this path structure for `output_dir` makes it compatible with `get_metrics.py`

for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
do
    python -m pl_finetune -c \
    t0/finetune.json+t0/${dataset}.json \
    -k \
    checkpoint=$PATH_TO_CHECKPOINT \
    output_dir=${PATH_TO_OUTPUT_DIR}/${dataset} \
    $*
done
