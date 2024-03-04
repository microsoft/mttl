from dataclasses import dataclass
import readline
import os
import glob
import sys
from typing import List

from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.evaluators.base import StoppingCriteriaSub, StoppingCriteriaList


def path_completer(text, state):
    # Expand ~ to the user's home directory
    text = os.path.expanduser(text)

    # Autocomplete directories with a wildcard
    if os.path.isdir(text):
        text += "*"

    # Return the next possible completion
    return (glob.glob(text) + [None])[state]


def setup_autocomplete():
    # Use the tab key for completion
    readline.parse_and_bind("tab: complete")

    # Set the completer function
    readline.set_completer(path_completer)


def generate(input, model, device):
    """Generate a sequence from a prompt."""
    import torch

    batch = model.tokenizer(input, return_tensors="pt")
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device)
    output_ids = model.generate(
        batch,
        max_new_tokens=200,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
        stopping_criteria=StoppingCriteriaList(
            [StoppingCriteriaSub(["\n\n"], model.tokenizer)]
        ),
    )

    output_ids = output_ids[0][batch["input_ids"].shape[1] :]
    return model.tokenizer.decode(output_ids, skip_special_tokens=True)


@dataclass
class Conversation:
    """A conversation between two agents."""

    prompts: List[str]
    responses: List[str]

    separator = "\n"
    prompt_template: str = "Instruct: {}\nAnswer:"
    response_template: str = " {}"

    def to_str(self):
        convo = ""
        for i in range(len(self.prompts)):
            if convo:
                convo += self.separator
            prompt = self.prompts[i]
            convo += self.prompt_template.format(prompt.strip())
            if len(self.responses) > i:
                response = self.responses[i]
                convo += self.response_template.format(response.strip())
        return convo

    def clear(self):
        self.prompts = []
        self.responses = []


@dataclass
class ConversationNoTemplate(Conversation):
    """A conversation between two agents."""

    prompt_template: str = "{}"


def main():
    from mttl.models.expert_config import ExpertConfig
    from mttl.models.expert_model import MultiExpertModel
    from mttl.datamodule.utils import get_tokenizer_with_args
    from mttl.utils import setup_logging
    import torch

    setup_autocomplete()
    setup_logging()

    config = ExpertConfig.parse(
        raise_error=False,
        c="./configs/wiki-mmlu/phi-2_flan_v3.json",
        extra_kwargs=dict(k.split("=") for k in sys.argv[1:]),
    )
    tokenizer = get_tokenizer_with_args(config.model, "gpt", "left", "left", True)
    model = MultiExpertModel(**vars(config), tokenizer=tokenizer).to("cuda")
    device = "cuda"

    print("Welcome to the LLM playground! Type 'exit' to leave.")

    conversation = (
        Conversation([], [])
        if config.use_instruct_template
        else ConversationNoTemplate([], [])
    )

    while True:
        try:
            print()

            expert_name = model.experts_names[0] if model.experts_names else None
            prompt = "{}>> ".format(f"({expert_name}) " if expert_name else "")

            user_input = input(prompt)

            if "%load" in user_input.lower():
                parts = user_input.partition("%load")[2].partition("from")
                module = parts[0].strip()
                library = parts[2].strip()
                model.delete_expert_container()
                expert = ExpertLibrary.get_expert_library(library)[module]
                model.add_expert_instance(expert, "default")
                continue

            if "%unload" in user_input.lower():
                model.delete_expert_container()
                continue

            if user_input.lower() in ["exit", "quit"]:
                break

            if user_input.lower() == "clear":
                conversation.clear()
                continue

            conversation.prompts.append(user_input)
            response = generate(conversation.to_str(), model, device)
            conversation.responses.append(response)

            print(response)
        except Exception as e:
            print.info("Error detected (%s). Type `exit` to leave." % e)


if __name__ == "__main__":
    main()
