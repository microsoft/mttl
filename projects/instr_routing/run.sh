python finetune_llama.py -c configs/llama/platypus/config_small.json \
    -k output_dir=/tmp/ \
    eval_superni=True \
    model=EleutherAI/gpt-neo-125m \
    dataset=platypus \
    eval_every=100 \
    max_input_length=4096 \
    n_skills=8 \
    xrouter_sim_metric=cosine \
    xrouting_option=4 \
    micro_batch_size=1 \
    train_batch_size=16 \
    precision=16 \
    predict_batch_size=32 \
    model_family=gpt