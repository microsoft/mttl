#!/bin/bash
conda activate comp_3.9
python /home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/src/evolution/transfer_matrix.py -c /home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/configs/wiki-mmlu/gpt2neo_1B_experts.json+/home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/configs/wiki-mmlu/gpt2neo_1B_dense.json \
        -k model_modifier=None \
        dataset=sordonia/flan-10k-flat \
        action=route \
        eval_metric=rougeL \
        truncation_side=left \
        finetune_task_name=FLAN_SUB19 \
        predict_batch_size=32 \
        output_dir=home/v-oostapenko/mttl_out \
        wandb_project=wiki_experts_ninja_eval_gpt2neo_1B \
        run_name=transfer_matrix_joint_training_1epoch \
        hf_repo_id=/home/v-oostapenko/dev/lucas_mttl/amlt/gptneo_1B_joint_flan19/gpt2_joint__3e-4_/EleutherAI_gpt-neo-1.3B-val/loss=0.6829.ckpt