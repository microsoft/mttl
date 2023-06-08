#!/bin/bash     


# alpaca lora tloen/alpaca-lora-7b o-shot    
# python eval/gen_ni_predictions.py --batch_size 5 --out_prefix "alpacaloratloen_peft7b16r[no_space]" --from_hf 3 --nshot 0 --model_name "yahma/llama-7b-hf"
# wait
# alpaca lora tloen/alpaca-lora-7b canonical 
# python eval/gen_ni_predictions.py --batch_size 2 --out_prefix "alpacaloratloen_peft7b[no_space]" --from_hf 3 --nshot 1 --model_name "yahma/llama-7b-hf"
# wait 
# python eval/gen_ni_predictions.py --batch_size 2 --out_prefix "llama_[no_space]" --from_hf 1 --nshot 1 --model_name "yahma/llama-7b-hf"

# python eval/eval_few_shot_ni.py --batch_size 2 --out_prefix "alpaca_full7b4r_alstsks" --from_hf 0 --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt" --nshot 1
# wait

       
# generate for my alpacas
# alpaca lora r4
# python eval/gen_ni_predictions.py --batch_size 2 --out_prefix "alpaca_full7b4r_[50]" --from_hf 0 --nshot 1 --model_name "yahma/llama-7b-hf" --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt" --n_tasks 50
# wait
# # # alpaca lora sc, average 
# python eval/gen_ni_predictions.py --batch_size 2 --out_prefix "sc_average_alpaca7b4r_atlas_[50]" --from_hf 0 --nshot 1  --skill_selector "average" --model_name "yahma/llama-7b-hf" --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt" --n_tasks 50
# wait
# # alpaca lora sc, topic
# python eval/gen_ni_predictions.py --batch_size 2 --out_prefix "sc_topic_alpaca7b4r_atlas_[50]" --skill_selector "topic" --from_hf 0 --nshot 1 --model_name "yahma/llama-7b-hf" --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt" --n_tasks 50
# wait
# # wait
# alpaca lora r4     
# python eval/gen_ni_predictions.py --batch_size 2 --out_prefix "alpaca_full7b4r_" --from_hf 0 --nshot 1 --model_name "yahma/llama-7b-hf" --n_tasks 20 --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt"
# wait
# alpaca lora sc, topic
# python eval/gen_ni_predictions.py --batch_size 2 --out_prefix "sc_topic_alpaca7b4r_sc_atlas_alstsks" --skill_selector "topic" --from_hf 0 --nshot 1 --model_name "yahma/llama-7b-hf" --n_tasks 20 --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt"
# wait
# # alpaca lora sc, average
# python eval/gen_ni_predictions.py --batch_size 2 --out_prefix "sc_average_alpaca7b4r_sc_atlas_alstsks" --skill_selector "average" --from_hf 0 --nshot 1 --model_name "yahma/llama-7b-hf" --n_tasks 20 --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt"

 
# llama adapter      
# python eval/gen_ni_predictions.py --batch_size 5 --out_prefix "alpaca_llama_adapter_[full]" --from_hf 0 --nshot 0 --model_name "yahma/llama-7b-hf" --model_path "/home/v-oostapenko/logs/llama_alpaca/llama_adapter/yahma_llama-7b-hfkrd9zi2k_llama_adapter-val/loss=0.6232.ckpt"
# wait
 
# python eval/gen_ni_predictions.py --batch_size 5 --out_prefix "pijama_7b_llama_adapter_[full]" --from_hf 0 --nshot 0 --usepijma_model_with_llama_adapter 1 --model_name "yahma/llama-7b-hf" --model_path "/home/v-oostapenko/logs/llama_alpaca/llama_adapter/yahma_llama-7b-hfkrd9zi2k_llama_adapter-val/loss=0.6232.ckpt"

 
# python eval/gen_ni_predictions.py --batch_size 5 --out_prefix "alpaca_full_[full]" --from_hf 1 --nshot 0 --usepijma_model_with_llama_adapter 0 --model_name "chavinlo/alpaca-native"
    
# python eval/gen_ni_predictions.py --batch_size 5 --out_prefix "alpaca_full4r_notrainonsoure_addeos[full,ptopt](oiia6ai8)" --from_hf 0 --nshot 0 --usepijma_model_with_llama_adapter 0 --model_name "yahma/llama-7b-hf" --model_path "/home/v-oostapenko/logs/llama_alpaca/lora_full_r4_3(notrainonsource)/yahma_llama-7b-hfoiia6ai8_alpaca_lora_full_r4-val/loss=0.4247.ckpt"
# wait                
python eval/gen_ni_predictions.py --batch_size 2 --skill_selector "topic" --out_prefix "alpaca_full4r_atlaslda_l1[full,ptopt,topic](9psyqia3)" --from_hf 0 --nshot 1 --usepijma_model_with_llama_adapter 0 --model_name "yahma/llama-7b-hf" --model_path "/home/v-oostapenko/logs/amlt_yahma_llama_atlas_9psyqia3_cluster_l2/alpaca4r_topic_ldal1/alpaca-lora_l2/best_model_alpaca_lora_atlas_cluster_te_ada_l2/loss=0.4305.ckpt"
# wait               
# python eval/gen_ni_predictions.py --batch_size 5 --skill_selector "average" --out_prefix "alpaca_full4r_atlaslda_l2[full,ptopt,average](9psyqia3)" --from_hf 0 --nshot 0 --usepijma_model_with_llama_adapter 0 --model_name "yahma/llama-7b-hf" --model_path "/home/v-oostapenko/logs/amlt_yahma_llama_atlas_9psyqia3_cluster_l2/alpaca4r_topic_ldal1/alpaca-lora_l2/best_model_alpaca_lora_atlas_cluster_te_ada_l2/loss=0.4305.ckpt"