export CONFIG_PATH=./projects/mhr/configs/
CUDA_VISIBLE_DEVICES=0 python projects/mhr/pl_train.py -c t0/xl-lm-adapt.json+t0/pretrain.json+t0/poly_lora.json -k eval_every=20_000 $*