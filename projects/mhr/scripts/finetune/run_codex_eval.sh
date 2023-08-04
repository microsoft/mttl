python -m inst_follow.eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 \
    --unbiased_sampling_size_n 5 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/llama_7B \
    --model_name_or_path llama_alpaca_finetune/best_model_alpaca_lora_atlas_cluster_8_ada/loss=0.5449.ckpt \
    --example_to_ids_path inst_follow/cluster_infos/atlas_by_instr_text-embedding-ada-002_ldalayer1.pkl \
    --eval_batch_size 32 