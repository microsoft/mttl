import os

base_dir = os.environ.get(
    "MODULES_DIR", "/home/v-oostapenko/dev/amlt/wiki_experts_train_per_cat_2_if/"
)
base_dir_tempalte = (
    lambda subject: f"{base_dir}/ll2_13b_expert_2_{subject}__qa-ostapeno_qa-openai_icl5_clen128_maxD-1_maxC8000_0_length_matched___5e-5_/{subject}/meta-llama_Llama-2-13b-hf-mmlu_test_oracle"
)

# MMLU_MODULES={
#     "formal_logic":
#     "machine_learning":
#     "global_facts":
#     "abstract_algebra":
#     "high_school_physics":
#     "college_biology":
#     "high_school_government_and_politics":
#     "prehistory":
#     "security_studies":
#     "sociology":
# }
