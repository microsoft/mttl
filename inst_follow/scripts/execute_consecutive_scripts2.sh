#!/bin/bash     
# wait              
#eval on vicuna average, but only ones with more than 100 points 
# python /home/v-oostapenko/dev/mttl/inst_follow/eval/get_answers_vicuna.py --skill_selector average --answer-file /home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answers_alpaca7b_lorar4_soft_cluster_atlas_average.jsonl
# wait 
# train a new soft clustering model with rank 8 (to be comperable )                      
# python /home/v-oostapenko/dev/mttl/inst_follow/finetune_allpaca_lora.py -c /home/v-oostapenko/dev/mttl/configs/llama/finetune_full_lora.json
# wait

# next 
# vicuna new model full 7b 4r
# python /home/v-oostapenko/dev/mttl/inst_follow/eval/get_answers_vicuna.py --answer-file /home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answers_alpaca_lora_full_7b4r.jsonl --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt"
# wait
# super-ni llama
# python eval/eval_few_shot_ni.py --batch_size 5 --out_prefix "llama" --from_hf 1
# wait


# super-ni new model
# python eval/eval_few_shot_ni.py --batch_size 5 --out_prefix "alpaca_full7b4r_alstsks" --from_hf 0 --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt"
# wait
# # super-ni  topic sc 4r                          
# python eval/eval_few_shot_ni.py --batch_size 5 --skill_selector "topic" --out_prefix  "sk_topic_alpaca7b4r_sc_atlas_alstsks" --from_hf 0 --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt"
# wait 
# # # super-ni  average sc 4r
# python eval/eval_few_shot_ni.py --batch_size 5 --skill_selector "average" --out_prefix "sk_average_alpaca7b4r_sc_atlas_alstsks" --from_hf 0 --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt"
# wait 

# # canonical    
# # super-ni llama
# python eval/eval_few_shot_ni.py --batch_size 2 --out_prefix "llama" --from_hf 1 --nshot 1
# wait
# # super-ni new model
# python eval/eval_few_shot_ni.py --batch_size 2 --out_prefix "alpaca_full7b4r_alstsks" --from_hf 0 --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt" --nshot 1
# wait
# # super-ni  topic sc 4r                          
# python eval/eval_few_shot_ni.py --batch_size 2 --skill_selector "topic" --out_prefix  "sk_topic_alpaca7b4r_sc_atlas_alstsks" --from_hf 0 --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt" --nshot 1
# wait 
# # # super-ni  average sc 4r
# python eval/eval_few_shot_ni.py --batch_size 2 --skill_selector "average" --out_prefix "sk_average_alpaca7b4r_sc_atlas_alstsks" --from_hf 0 --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt" --nshot 1
# # wait 
 
# /home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt --- alpaca full 7b 4r

# /home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt -- alpaca 7b 4r sc atlas





# # vicuna evals
# # alpaca lora full 7b 4r
# python /home/v-oostapenko/dev/mttl/inst_follow/eval/get_gpt_review_vicuna.py --output-review-file "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/review/gpt35_vs_alpacalorafull7b4r_bygpt4.jsonl" --answer-file-list "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answer_gpt35.jsonl" "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answers_alpaca_lora_full_7b4r.jsonl"
# wait
# # alpaca lora topic 7b 4r
# python /home/v-oostapenko/dev/mttl/inst_follow/eval/get_gpt_review_vicuna.py --output-review-file "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/review/gpt35_vs_alpacaloraatlas7b4rtopic_bygpt4.jsonl" --answer-file-list "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answer_gpt35.jsonl" "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answers_alpaca7b_lorar4_soft_cluster_atlas_topic.jsonl"
# wait
# # alpaca lora average 7b 4r
# python /home/v-oostapenko/dev/mttl/inst_follow/eval/get_gpt_review_vicuna.py --output-review-file "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/review/gpt35_vs_alpacaloraatlas7b4raverage_bygpt4.jsonl" --answer-file-list "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answer_gpt35.jsonl" "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answers_alpaca7b_lorar4_soft_cluster_atlas_average.jsonl"
# # wait






# # vicuna old
# # alpaca lora full 7b 4r
# python /home/v-oostapenko/dev/mttl/inst_follow/eval/get_gpt_review_vicuna.py --output-review-file "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/review/gpt35_vs_[old]alpacalorafull7b8r_bygpt4.jsonl" --answer-file-list "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answer_gpt35.jsonl" "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/[old]answers_alpaca_lora_full_7b8r.jsonl"
# wait
# # alpaca lora topic 7b 4r
# python /home/v-oostapenko/dev/mttl/inst_follow/eval/get_gpt_review_vicuna.py --output-review-file "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/review/gpt35_vs_[old]alpacaloraatlas7b4rtopic_bygpt4.jsonl" --answer-file-list "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answer_gpt35.jsonl" "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/[old]answers_alpaca7b_lorar4_soft_cluster_atlas_average.jsonl"
# wait
# # alpaca lora average 7b 4r
# python /home/v-oostapenko/dev/mttl/inst_follow/eval/get_gpt_review_vicuna.py --output-review-file "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/review/gpt35_vs_[old]alpacaloraatlas7b4raverage_bygpt4.jsonl" --answer-file-list "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answer_gpt35.jsonl" "/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/[old]answers_alpaca7b_lorar4_soft_cluster_atlas_average.jsonl"
# wait
    
# train alpaca lora topic 7b 4r with clustering             
python /home/v-oostapenko/dev/mttl/finetune_llama.py -c /home/v-oostapenko/dev/mttl/configs/longform/finetune_full_lora.json
# wait 
# # train alpaca lora full 7b 4r
# python /home/v-oostapenko/dev/mttl/finetune_llama.py -c /home/v-oostapenko/dev/mttl/configs/llama/finetune_full_lora.json 