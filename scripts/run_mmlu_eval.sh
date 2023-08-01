# CUDA_VISIBLE_DEVICES=1 nohup python -m inst_follow.eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/alpaca-lora-7B-0shot/ \
#     --eval_batch_size 2 \
#     --model_name_or_path tloen/alpaca-lora-7b \
#     --tokenizer_name_or_path yahma/llama-7b-hf \
#     > test_mmlu_eval.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -m inst_follow.eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/cluster-lora-7B-0shot/ \
#     --eval_batch_size 2 \
#     --model_name_or_path tloen/alpaca-lora-7b \
#     --tokenizer_name_or_path yahma/llama-7b-hf \
#     --model_name_or_path llama_alpaca_finetune/best_model_alpaca_lora_atlas_cluster_8_ada/loss=0.5449.ckpt \
#     --example_to_ids_path inst_follow/cluster_infos/atlas_by_instr_text-embedding-ada-002_ldalayer1.pkl \
#     --tokenizer_name_or_path yahma/llama-7b-hf \
#     --skill_selector topic \
#     > test_mmlu_topic_routing8_eval.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -m inst_follow.eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/polymu-lora-7B-0shot/ \
#     --eval_batch_size 2 \
#     --model_name_or_path tloen/alpaca-lora-7b \
#     --tokenizer_name_or_path yahma/llama-7b-hf \
#     --model_name_or_path llama_alpaca_finetune/poly_mu_alpaca/loss=0.5422.ckpt \
#     --example_to_ids_path inst_follow/cluster_infos/atlas_by_instr_text-embedding-ada-002_ldalayer1.pkl \
#     --tokenizer_name_or_path yahma/llama-7b-hf \
#     --skill_selector poly \
#     > test_mmlu_polymu_routing8_eval.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -m inst_follow.eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/topic_average-lora-7B-0shot/ \
    --eval_batch_size 2 \
    --model_name_or_path tloen/alpaca-lora-7b \
    --tokenizer_name_or_path yahma/llama-7b-hf \
    --model_name_or_path llama_alpaca_finetune/best_model_alpaca_lora_atlas_cluster_8_ada/loss=0.5449.ckpt \
    --example_to_ids_path inst_follow/cluster_infos/atlas_by_instr_text-embedding-ada-002_ldalayer1.pkl \
    --tokenizer_name_or_path yahma/llama-7b-hf \
    --skill_selector average \
    > test_mmlu_topic_average_routing8_eval.log 2>&1 &