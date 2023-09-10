/bin/rm -rf /tmp/runs/alpaca/dense/
python finetune_llama.py \
    -c configs/alpaca/gptneo_125m.json+configs/alpaca/gptneo_125m_dense.json \
    -k output_dir=/tmp/runs/alpaca/dense/ \
    dataset=alpaca \
    eval_mmlu=True \
    eval_superni=True \
    total_steps=1000
