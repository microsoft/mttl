#!/bin/bash     

# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval_llama.py -c /home/v-oostapenko/dev/mttl/configs/llama/finetune_full.json --model-path "/home/v-oostapenko/logs/llama_alpaca/lora_full/best_model/loss=0.5940.ckpt" --skill_selector none
# wait
# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval_llama.py -c /home/v-oostapenko/dev/mttl/configs/llama/finetune_atlas_cluster_by_inst.json --model-path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/best_model/loss=0.6022.ckpt" --skill_selector average --answer-file /home/v-oostapenko/dev/mttl/compositional_adapters/eval/vicuna_questions/table/answer/answers_alpaca7b_lorar4_soft_cluster_atlas_average.jsonl
# wait 
# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval_llama.py -c /home/v-oostapenko/dev/mttl/configs/llama/finetune_atlas_cluster_by_inst.json --model-path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/best_model/loss=0.6022.ckpt" --skill_selector topic --answer-file /home/v-oostapenko/dev/mttl/compositional_adapters/eval/vicuna_questions/table/answer/answers_alpaca7b_lorar4_soft_cluster_atlas_topic.jsonl 
# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval/eval_gpt_review.py -q /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/question.jsonl -a /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/answer_gpt35.jsonl /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/answers_alpaca_lora_full_7b.jsonl -p /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/prompt.jsonl -r /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/reviewer.jsonl -o /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/review/gpt35/gpt25_vs_answers_alpaca_lora_full_7b.jsonl
# wait     
# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval/eval_gpt_review.py -q /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/question.jsonl -a /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/answers_alpaca_lora_full_7b.jsonl /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/answers_alpaca7b_lorar4_soft_cluster_atlas_average.jsonl -p /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/prompt.jsonl -r /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/reviewer.jsonl -o /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/review/gpt35/answers_alpaca_lora_full_7b_vs_answers_alpaca7b_lorar4_soft_cluster_atlas_average.jsonl
# wait     
# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval/eval_gpt_review.py -q /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/question.jsonl -a /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/answer_gpt35.jsonl /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/answers_alpaca7b_lorar4_soft_cluster_atlas_topic.jsonl -p /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/prompt.jsonl -r /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/reviewer.jsonl -o /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/review/gpt35/answer_gpt35_vs_answers_alpaca7b_lorar4_soft_cluster_atlas_topic.jsonl

      
# original llama (no finetuning)
python /home/v-oostapenko/dev/mttl/compositional_adapters/eval_get_answers.py --model-id "llama" -c "/home/v-oostapenko/dev/mttl/configs/llama/eval_configs/eval_llama.json" --destination "/home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/self_instruct/goot_stuff/llama/answers_llama7b_on_si_train[si_prpt][sc_GPT_vs_ground_truth][sc_rouge_vs_ground_truth][cleaned_response][new_answers].jsonl" --n_examples_per_topic 10 --n_topics 10
# wait 
# # generate responses by clustered poly_lora
# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval_get_answers.py  --model-id "alpaca" --model-path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt" -c /home/v-oostapenko/dev/mttl/configs/llama/finetune_atlas_cluster_by_inst.json --destination "/home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/self_instruct/ni_answers_alpaca7b_lorar4_soft_cluster_atlas_on_si_train.jsonl" --n_examples_per_topic 10 --n_topics 10
# wait
               
# add eval scores against GT     ~    
# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval_get_gpt_score.py --answers_file "/home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/self_instruct/si_answers_llama7b_on_si_train[no sampling].json"
# wait
# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval_get_gpt_score.py --answers_file /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/self_instruct/si_answers_alpaca7b_lorar4_soft_cluster_atlas_on_si_train.jsonl

# add eval scores against GPT_3.5
# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval_get_gpt_score.py --answers_file /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/self_instruct/si_response_alpaca7b_lorar4_soft_cluster_atlas_on_si_train.jsonl --gt_answers /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/self_instruct/si_response_alpaca7b_lorar4_soft_cluster_atlas_on_si_train.jsonl
# wait
# python /home/v-oostapenko/dev/mttl/compositional_adapters/eval_get_gpt_score.py --answers_file /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/self_instruct/si_response_llama7b_on_si_train.jsonl --gt_answers /home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/self_instruct/si_response_alpaca7b_lorar4_soft_cluster_atlas_on_si_train.jsonl
