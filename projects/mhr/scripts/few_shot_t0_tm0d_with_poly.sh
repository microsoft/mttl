
for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
do
    python -m pl_finetune -c \
    t0/finetune.json+t0/${dataset}.json \
    -k \
    checkpoint=${AMLT_MAP_INPUT_DIR} \
    output_dir=${AMLT_OUTPUT_DIR}/${dataset} \
    -k model_modifier=poly_lora \
    -k poly_selector=poly
done
