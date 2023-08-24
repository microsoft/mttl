from finetune_llama import RoutingConfig
from transformers import AutoModelForCausalLM
from eval.ni.gen_ni_predictions import eval_ni


if __name__ == "__main__":
    config = RoutingConfig.parse()

    config.model = "uoe-nlp/gpt-neo-125m_instruction-tuned_sni"
    config.model_family = "gpt"
    config.data_dir = "/datadrive2/sni/"
    config.max_input_length = 1024
    config.max_output_length = 128

    model = AutoModelForCausalLM.from_pretrained(config.model)
    model = model.cuda()

    print(eval_ni(config, model, nshot=2))
