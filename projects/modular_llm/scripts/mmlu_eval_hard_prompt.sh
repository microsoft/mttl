export MMLU_CONFIG=""

python mmlu_eval_experts.py -c wiki-mmlu/phi-2_flan.json \
    -k output_dir=results/mmlu_with_hard_prompt \
    -k mmlu_use_hard_prompt="The following question is about biology. Remember what you learned, answer with A, B, C, D only." \
    -k finetune_task_name=college_biology
