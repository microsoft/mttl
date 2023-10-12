
CUDA_VISIBLE_DEVICES=0 python train_experts.py \
    -c wiki-mmlu/gptneo_125m_experts.json+wiki-mmlu/gptneo_125m_dense.json \
    -k output_dir=gptneo_125m_experts/infollow/ \
    dataset=inverse-oa1 \
    total_steps=100 \
    warmup_proportion=0.1 \
    tensorboard=True \
    precision=16-mixed
