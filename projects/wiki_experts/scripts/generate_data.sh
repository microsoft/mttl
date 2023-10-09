
# conda activate comp_3.9_pl2
#open AI
# python /home/v-oostapenko/dev/mttl/projects/wiki_experts/cli_qa_creator.py geni --model-path gpt-35-turbo --output-filename=/home/v-oostapenko/mttl_out/generated_gpt35_turbo_icl5.jsonl

# invese platypus instructions icl 5
python /home/v-oostapenko/dev/mttl/projects/wiki_experts/cli_qa_creator.py geni --model-path sordonia/llama2-13b-platypus-inverse --output-filename /home/v-oostapenko/mttl_out/generated_llama2_13b_platypus_inverse_icl_5.jsonl
wait
python /home/v-oostapenko/dev/mttl/projects/wiki_experts/cli_qa_creator.py gena --model-path sordonia/llama2-13b-platypus --instruction-json /home/v-oostapenko/mttl_out/generated_llama2_13b_platypus_inverse_icl_5.jsonl
wait

python /home/v-oostapenko/dev/mttl/projects/wiki_experts/cli_qa_creator.py geni --n_icl 0 --model-path sordonia/llama2-13b-platypus-inverse --output-filename /home/v-oostapenko/mttl_out/generated_llama2_13b_platypus_inverse_icl_0.jsonl
wait
python /home/v-oostapenko/dev/mttl/projects/wiki_experts/cli_qa_creator.py gena --model-path sordonia/llama2-13b-platypus --instruction-json /home/v-oostapenko/mttl_out/generated_llama2_13b_platypus_inverse_icl_0.jsonl