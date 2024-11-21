from mttl.models.library.expert import load_expert

# expert = load_expert(
#     "/home/mila/z/zhan.su/.cache/huggingface/hub/models--pclucas14--library-phi3-4k_5ep-fixed/snapshots/a8064a0322d647bc56cd0a25c514b8d412e6b129/wmt14_translate_fr_en_1_0_0.ckpt"
# )

expert = load_expert(
    "/home/mila/z/zhan.su/code/mttl/projects/modular_llm/gptneo125m_debug/best_mode_min_metric_val-loss_value_2.9076_step_500.ckpt"
)

breakpoint()
