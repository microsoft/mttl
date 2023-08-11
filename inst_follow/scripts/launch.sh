#!/bin/bash
conda activate comp_3.9   
export WANDB_API_KEY=5471c6f5e95b83a3c0b2072b0614eb67c2f23eab
cd /home/v-oostapenko/dev/mttl/
nohup python /home/v-oostapenko/dev/mttl/finetune_llama.py -c /home/v-oostapenko/dev/mttl/configs/llama/finetune_atlas_cluster_by_inst_local.json > /home/v-oostapenko/dev/mttl/inst_follow/outputs/output_alpaca_shared_a.txt &