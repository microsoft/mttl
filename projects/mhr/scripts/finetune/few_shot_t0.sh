
for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
do
    python -m pl_finetune -c \
    t0/finetune.json+t0/${dataset}.json+t0/t5s.json \
    -k \
    checkpoint=${CHECKPOINT_DIR} \
    output_dir=${OUTPUT_DIR}/${dataset} \
    $*
done
