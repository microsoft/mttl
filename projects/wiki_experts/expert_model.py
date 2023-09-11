import torch

from mttl.models.modifiers.routing import RoutingInfo
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from mttl.models.utils import (
    EfficientCheckpointModule,
)

from mttl.models.utils import convert_and_push_to_hub, download_from_hub
from mttl.models.modifiers.experts import add_expert_to_transformer
from mttl.utils import get_checkpoint_path, logger
from config import ExpertConfig


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


class MultiExpertModel(EfficientCheckpointModule):
    def __init__(self, **kwargs):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters(ignore=["tokenizer", "model_object"])

        self.tokenizer = kwargs["tokenizer"]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.model: AutoModelForCausalLM = None

        if "llama" in self.hparams.model:
            model_object = LlamaForCausalLM.from_pretrained(
                self.hparams.model,
                load_in_8bit=self.hparams.load_in_8bit,
                torch_dtype=torch.float32,
                device_map="auto",
            )
        else:
            model_object = AutoModelForCausalLM.from_pretrained(self.hparams.model)

        if model_object.config.vocab_size != len(self.tokenizer):
            model_object.resize_token_embeddings(len(self.tokenizer))

        if self.hparams.load_in_8bit:
            model_object = prepare_model_for_kbit_training(model_object)

        self.model = model_object
        self.experts = []

    def load_expert(
        self,
        expert_path: str,
        expert_name: str = None,
        action: str = "merge",
        is_default: bool = False,
    ):
        # load the expert weights
        import json
        import os

        if os.path.isdir(expert_path):
            expert_checkpoint = get_checkpoint_path(expert_path)
        else:
            expert_checkpoint = download_from_hub(expert_path)

        logger.info(f"Loading expert from {expert_checkpoint}...")

        expert_checkpoint = torch.load(expert_checkpoint, map_location="cpu")

        expert_config = ExpertConfig(
            kwargs=expert_checkpoint["hyper_parameters"], silent=True, raise_error=False
        )
        if expert_config.expert_name is None:
            logger.info("Assigning expert name, not found in checkpoint: {}".format(expert_path))
            expert_name = os.path.basename(expert_path)

        expert_weights = expert_checkpoint["state_dict"]
        expert_weights = {
            k.replace("model.", "", 1): v for k, v in expert_weights.items()
        }
        if self.hparams.model != expert_config.model:
            raise ValueError(
                "The expert has been trained on top of a different model!"
                " Detected: {} - Expected: {}".format(
                    expert_config.model, self.hparams.model
                )
            )

        logger.info(
            f"Adding expert with name {expert_name}... with action ... {action}!"
        )

        self.model = add_expert_to_transformer(
            self.model,
            expert_name,
            expert_config,
            expert_weights,
            action=action,
            is_default=is_default,
        )
        if action != "merge":
            self.experts.append(expert_name)

    @property
    def generation_config(self):
        return self.model.generation_config

    def expert_choice(self, batch, **kwargs):
        input_ids = batch["input_ids"]
        mask = batch["input_ids"].ne(self.tokenizer.pad_token_id)

        # convert left to right padding here
        def roll_along(arr, shifts, dim):
            assert arr.ndim - 1 == shifts.ndim
            dim %= arr.ndim
            shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
            dim_indices = torch.arange(arr.shape[dim]).reshape(shape).to(arr.device)
            indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
            return torch.gather(arr, dim, indices)

        input_ids = roll_along(input_ids, mask.sum(1), 1)
        mask = input_ids.ne(0)
        labels = torch.masked_fill(input_ids, ~mask, -100)

        scores = []
        for expert in self.experts:
            batch["task_names"] = [expert for _ in range(batch["input_ids"].shape[0])]
            self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(
                batch
            )
            outputs = self.model.forward(
                input_ids,
                attention_mask=mask,
            )
            # calculate loss, could also be done inside of the model
            bs = input_ids.size(0)
            logits = outputs.logits
            vocab_size = logits.size(-1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss.view((bs, -1)).sum(1)
            # mean only non-zero
            scores.append(loss.cpu())

        scores = torch.stack(scores, 0)
        expert_indices = scores.argmin(0)
        return [self.experts[i] for i in expert_indices]

    def generate(
        self,
        batch,
        **kwargs,
    ):
        if getattr(self.hparams, 'experts_auto_route', False):
            logger.info("Auto-routing... ground-truth tasks: {}".format(batch["task_names"]))
            batch["task_names"] = self.expert_choice(batch)
            logger.info("Auto-route tasks: {}".format(batch["task_names"]))

        if hasattr(self.model, "task_id_container"):
            self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(
                batch
            )

        generations = self.model.generate(inputs=batch["input_ids"], **kwargs)
        return generations
