for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
# for dataset in storycloze
do
    CUDA_VISIBLE_DEVICES=1 python -m pl_finetune -c \
    t0/finetune.json+t0/${dataset}.json \
    -k \
    checkpoint=pretrain_poly_lora \
    output_dir=finetune_poly_lora_freeze_skill/${dataset} \
    finetune_type="Z"
done
