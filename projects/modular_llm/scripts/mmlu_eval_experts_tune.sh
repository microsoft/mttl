export MMLU_CONFIG=""
SUB_10=("abstract_algebra" "global_facts" "college_biology" "formal_logic" "global_facts" "high_school_government_and_politics" "high_school_physics" "machine_learning" "prehistory" "security_studies" "sociology")

for task in ${SUB_10[@]}; do
for w1 in "0.0" "1.0" "2.0"; do
for w2 in "0.0" "1.0" "2.0"; do

module_graph="${task} -> linear(../../amlt/platypus/platypus-13b-right/:${w1}, ../../amlt/mmlu_experts_platypus_qa/ll2-13b-expert-${task}/${task}/:${w2})"
python mmlu_eval_experts.py -c wiki-mmlu/llama2_13b_experts.json \
    -k load_in_8bit=False finetune_task_name=${task} output_dir=results/${task}/tuning/${w1}-${w2}/valid/ \
    -k module_graph="${task} -> linear(../../amlt/platypus/platypus-13b-right/:${w1}, ../../amlt/mmlu_experts_platypus_qa/ll2-13b-expert-${task}/${task}/:${w2})" \
    -k mmlu_test_split=validation

python mmlu_eval_experts.py -c wiki-mmlu/llama2_13b_experts.json \
    -k load_in_8bit=False finetune_task_name=${task} output_dir=results/${task}/tuning/${w1}-${w2}/test/ \
    -k module_graph="${task} -> linear(../../amlt/platypus/platypus-13b-right/:${w1}, ../../amlt/mmlu_experts_platypus_qa/ll2-13b-expert-${task}/${task}/:${w2})"
    echo ${module_graph} >> results/${task}/tuning/${w1}-${w2}/module_graph.txt
done
done
done
