#!/bin/bash
conda activate comp_3.9
# phi 20 joint 4 epochs
python /home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/src/evolution/transfer_matrix.py -c /home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/configs/wiki-mmlu/gpt2neo_1B_experts.json+/home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/configs/wiki-mmlu/gpt2neo_1B_dense.json \
        -k model_modifier=None \
        dataset=sordonia/flan-10k-flat \
        action=route \
        eval_metric=rougeL \
        truncation_side=left \
        finetune_task_name=FLAN_PHI_20 \
        predict_batch_size=32 \
        output_dir=/home/v-oostapenko/mttl_out \
        wandb_project=wiki_experts_ninja_eval_gpt2neo_1B \
        run_name=transfer_matrix_joint_4_epochs_flan_phi20_tasks \
        hf_repo_id=/home/v-oostapenko/dev/lucas_mttl/amlt/gptneo_1B_joint_flan_phi20_runs/joint_4_epochs_flan_phi20_tasks/EleutherAI_gpt-neo-1.3B-val/loss=0.8855.ckpt &
        
wait

# flan sub 19 joint 4 epochs
python /home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/src/evolution/transfer_matrix.py -c /home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/configs/wiki-mmlu/gpt2neo_1B_experts.json+/home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/configs/wiki-mmlu/gpt2neo_1B_dense.json \
        -k model_modifier=None \
        dataset=sordonia/flan-10k-flat \
        action=route \
        eval_metric=rougeL \
        truncation_side=left \
        finetune_task_name=FLAN_SUB19 \
        predict_batch_size=32 \
        output_dir=/home/v-oostapenko/mttl_out \
        wandb_project=wiki_experts_ninja_eval_gpt2neo_1B \
        run_name=tranfer_matrix_joint_4epochs_flan_sub19 \
        hf_repo_id=/home/v-oostapenko/dev/lucas_mttl/amlt/gptneo_1B_joint_flan19_runs/joint_4_epochs/EleutherAI_gpt-neo-1.3B-val/loss=0.6114.ckpt &

wait

python /home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/src/evolution/transfer_matrix.py -c /home/v-oostapenko/dev/lucas_mttl/projects/wiki_experts/configs/wiki-mmlu/phi-2_flan.json \
        -k model_modifier=None \
        dataset=sordonia/flan-10k-flat \
        action=route \
        eval_metric=rougeL \
        truncation_side=left \
        finetune_task_name=FLAN_PHI_20 \
        predict_batch_size=32 \
        output_dir=/home/v-oostapenko/mttl_out \
        wandb_project=wiki_experts_ninja_eval_gpt2neo_1B \
        run_name=transfer_matrix_joint_4_epochs_phi \
        hf_repo_id=/home/v-oostapenko/dev/lucas_mttl/amlt/phi2_train_flan_experts_jointruns/joint_phi_4_epochs_3e-4/phi-2-val/loss=1.0514.ckpt