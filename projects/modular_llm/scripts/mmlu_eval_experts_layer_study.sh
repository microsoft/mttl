export MMLU_CONFIG=""

# CUDA_VISIBLE_DEVICES=0 python mmlu_eval_experts.py -c wiki-mmlu/llama2_13b.json \
#    -k load_in_8bit=False finetune_task_name=college_biology output_dir=results/llama2_cb

for task in "college_biology" "global_facts"; do
CUDA_VISIBLE_DEVICES=0 python mmlu_eval_experts.py -c wiki-mmlu/llama2_13b_experts.json \
    -k load_in_8bit=False finetune_task_name=${task} output_dir=results/${task}/-${layer_num}/r1_wd0.01 \
    load_module=../../amlt/mmlu_wiki_experts_r1_wd0.01/ll2-13b-expert-${task}/${task}/:route:-${layer_num}
done

for layer_num in 0 5 9 15 39; do
CUDA_VISIBLE_DEVICES=0 python mmlu_eval_experts.py -c wiki-mmlu/llama2_13b_experts.json \
    -k load_in_8bit=False finetune_task_name=${task} output_dir=results/${task}/${layer_num}-/r1_wd0.01 \
    load_module=../../amlt/mmlu_wiki_experts_r1_wd0.01/ll2-13b-expert-${task}/${task}/:route:${layer_num}-
done
done
