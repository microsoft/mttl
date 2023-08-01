export CUDA_VISIBLE_DEVICES=0
python -m inst_follow.eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/llama-7B-no-cot/ \
    --model_name_or_path llama_alpaca_finetune/poly_mu_alpaca/loss=0.5422.ckpt \
    --tokenizer_name_or_path yahma/llama-7b-hf \
    --example_to_ids_path inst_follow/cluster_infos/atlas_by_instr_text-embedding-ada-002_ldalayer1.pkl \
    --eval_batch_size 10 \
    --max_num_examples_per_task 40 \
    --load_from llama \
    --no_cot \
    --use_chat_format