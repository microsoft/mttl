
CUDA_VISIBLE_DEVICES=0 python mmlu_eval_experts.py \
    -c wiki-mmlu/gptneo_125m.json+wiki-mmlu/gptneo_125m_dense.json \
    -k output_dir=gptneo_125m_experts/mmlu_eval/ \
    $*
