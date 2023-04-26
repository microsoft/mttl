# for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
for dataset in wic storycloze wsc
do
    CUDA_VISIBLE_DEVICES=1 python -m pl_finetune -c \
    t0/finetune.json+t0/${dataset}.json \
    -k \
    checkpoint=pretrain_tensororderpoly_lora_order_4 \
    output_dir=finetune_tensororderpoly_lora_order_4/${dataset}
done
