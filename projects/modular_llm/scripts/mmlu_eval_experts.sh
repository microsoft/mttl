export MMLU_CONFIG=""

for task in "abstract_algebra" "global_facts" "college_biology" "formal_logic" "global_facts" "high_school_government_and_politics" "high_school_physics" "machine_learning" "prehistory" "security_studies" "sociology"; do

python mmlu_eval_experts.py -c wiki-mmlu/llama2_13b_experts.json \
    -k load_in_8bit=False finetune_task_name=${task} output_dir=results/${task}/baseline

python mmlu_eval_experts.py -c wiki-mmlu/llama2_13b_experts.json \
    -k load_in_8bit=False finetune_task_name=${task} output_dir=results/${task}/expert \
    -k load_module=../../amlt/mmlu_experts_platypus_qa/ll2-13b-expert-${task}/${task}/:route

python mmlu_eval_experts.py -c wiki-mmlu/llama2_13b_experts.json \
    -k load_in_8bit=False finetune_task_name=${task} output_dir=results/${task}/platypus \
    -k load_module=../../amlt/platypus/platypus-13b-right/:merge

python mmlu_eval_experts.py -c wiki-mmlu/llama2_13b_experts.json \
    -k load_in_8bit=False finetune_task_name=${task} output_dir=results/${task}/expert+platypus \
    -k load_module=../../amlt/platypus/platypus-13b-right/=platypus:merge \
    -k load_module=../../amlt/mmlu_experts_platypus_qa/ll2-13b-expert-${task}/${task}/:route

done
