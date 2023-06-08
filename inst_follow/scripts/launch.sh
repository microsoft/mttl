#!/bin/bash
conda activate comp_3.8
cd ..
nohup python /home/v-oostapenko/dev/mttl/compositional_adapters/finetune_allpaca_lora.py -c /home/v-oostapenko/dev/mttl/configs/llama/finetune_atlas_cluster_by_inst.json > output_atlas.txt & olau