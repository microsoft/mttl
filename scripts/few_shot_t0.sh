for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
do
    python -m pl_finetune -c \
    t0/finetune.json+t0/${dataset}.json \
    -k \
    checkpoint=pretrain_tensorpoly \
    output_dir=finetune/${dataset}
done
