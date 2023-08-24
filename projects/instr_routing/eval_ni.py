from finetune_llama import RoutingConfig
from transformers import AutoModelForCausalLM
from eval.ni.gen_ni_predictions import eval_ni


if __name__ == "__main__":
    config = RoutingConfig.parse()

    config.model = "EleutherAI/gpt-neo-125m"
    config.data_dir = "/datadrive2/sni/"

    model = AutoModelForCausalLM.from_pretrained(config.model)
    model = model.cuda()

    print(eval_ni(config, model, nshot=2))
