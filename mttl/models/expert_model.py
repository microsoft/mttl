import math
import re
import threading
from collections import defaultdict
from functools import partial
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedModel

from mttl.models.containers import add_expert_to_transformer
from mttl.models.containers.expert_containers import ExpertContainer
from mttl.models.containers.selectors import (
    ArrowSelectorConfig,
    Selector,
    SelectorConfig,
)
from mttl.models.expert_config import ExpertConfig
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.library_transforms import (
    ArrowConfig,
    HiddenStateComputerConfig,
)
from mttl.models.llama_patch import replace_attn_with_flash_attn
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.base import ModifierConfig
from mttl.models.modifiers.lora import SkilledLoRAConfig
from mttl.models.modifiers.routing import RoutingInfo
from mttl.models.utils import EfficientCheckpointModule, prepare_model_for_kbit_training
from mttl.utils import logger

torch.set_float32_matmul_precision("high")


class ExpertModel(EfficientCheckpointModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = kwargs.pop("tokenizer", None)
        model_object = kwargs.pop("model_object", None)

        # log hyperparameters
        self.save_hyperparameters(kwargs)

        self.load_in_4bit = kwargs.get("load_in_4bit", None) or False
        self.load_in_8bit = kwargs.get("load_in_8bit", None) or False
        self.model: PreTrainedModel = None
        self.accumulate_metrics_batch = defaultdict(list)

        if model_object is None:
            from mttl.models.utils import model_loader_helper

            model_object = model_loader_helper(
                self.hparams.model,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                device_map=getattr(self.hparams, "device_map", "cpu"),
            )

        if self.load_in_8bit:
            model_object = prepare_model_for_kbit_training(model_object)

        # rebuild the training config, a bit cumbersome, but that's life
        self.training_config = ExpertConfig.fromdict(kwargs)
        self.training_config.vocab_size = (
            model_object.get_input_embeddings().num_embeddings
        )

        # init the transformer just with the modifier config, this avoids
        # passing the whole training config to the modify_transformer func
        self.modifier_config = ModifierConfig.from_training_config(self.training_config)
        # config about the routing
        if "selector_config" in kwargs:
            self.selector_config = kwargs.pop("selector_config")
        else:
            self.selector_config = SelectorConfig.from_training_config(
                self.training_config
            )

        self.model = modify_transformer(model_object, self.modifier_config)

        # replace w flash attn!
        replace_attn_with_flash_attn(self.model)

        self.test_results = []
        self.best_val_result = None
        self._inference_outputs = []
        self._log_pref = kwargs.get("logging_prefix", "")

    def forward(self, batch, reduction="mean"):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        self.set_routing_infos(batch)

        outputs = self.model.forward(input_ids, attention_mask=batch["attention_mask"])

        # calculate loss, could also be done inside of the model
        bs = input_ids.size(0)
        logits = outputs.logits
        vocab_size = logits.size(-1)
        labels = labels.squeeze(-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        # reshape back
        if reduction == "none":
            loss = loss.view((bs, -1))
            # mean only non-zero
            non_zero_loss = (loss != 0).sum(dim=-1)
            non_zero_loss[non_zero_loss == 0] = 1
            loss = loss.sum(dim=-1) / non_zero_loss

        del outputs, shift_logits, shift_labels
        return loss

    def compute_unlikelihood_loss(self, batch, reduction="mean"):
        # compute lm losses for all the options
        loss = self.forward(batch, reduction="none")

        num = 0
        lm_loss, mc_loss = [], []
        for i, num_options in enumerate(batch["num_options"]):
            loss_slice = loss[num : num + num_options]
            lm_loss.append(loss_slice[batch["labels_index"][i]])
            mc_loss.append(
                -torch.nn.functional.log_softmax(-loss_slice, dim=0)[
                    batch["labels_index"][i]
                ]
            )
            num += num_options

        lm_loss = torch.stack(lm_loss)
        mc_loss = torch.stack(mc_loss)
        if reduction == "mean":
            lm_loss = lm_loss.mean()
            mc_loss = mc_loss.mean()
        return lm_loss, mc_loss

    def training_step(self, batch, _):
        if "num_options" in batch:
            loss, mc_loss = self.compute_unlikelihood_loss(batch)
            self.log(
                f"{self._log_pref}train/mc_loss", loss, on_step=True, prog_bar=True
            )
            total_loss = loss + mc_loss
        else:
            loss = self.forward(batch)
            total_loss = loss

        self.log(f"{self._log_pref}train/loss", loss, on_step=True, prog_bar=True)
        self.log(
            f"{self._log_pref}train/total_loss", total_loss, on_step=True, prog_bar=True
        )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self._inference_outputs.clear()

    def on_test_epoch_start(self) -> None:
        self._inference_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self.log_loss(split="val")

    def on_test_epoch_end(self) -> None:
        self.log_loss(split="test")

    def test_step(self, batch, batch_idx):
        if "num_options" in batch:
            loss, _ = self.compute_unlikelihood_loss(batch, reduction="none")
        else:
            loss = self.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self._inference_outputs += [(loss.detach(),)]
        return mean_loss

    def get_loss_for_all(self, batch, batch_idx):
        loss = self.forward(batch, reduction="none")
        return loss

    def validation_step(self, batch, batch_idx):
        if "num_options" in batch:
            loss, _ = self.compute_unlikelihood_loss(batch, reduction="none")
        else:
            loss = self.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self._inference_outputs += [(loss.detach(),)]
        return mean_loss

    def log_loss(self, split="val"):
        outputs = self._inference_outputs
        losses = torch.cat([out[0] for out in outputs], 0)
        self._inference_outputs.clear()
        self.log(
            f"{self._log_pref}{split}/loss",
            losses.mean(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # log also the best val/loss sofar
        if split == "val":
            if self.best_val_result is None:
                self.best_val_result = losses.mean()
            else:
                if losses.mean() < self.best_val_result:
                    self.best_val_result = losses.mean()
                    self.log(
                        f"{self._log_pref}{split}/best_loss",
                        losses.mean(),
                        on_epoch=True,
                        prog_bar=True,
                    )

    def set_routing_infos(self, batch, generate=False):
        self.model.info_container["routing_infos"] = RoutingInfo.from_batch(batch)

    @property
    def generation_config(self):
        return self.model.generation_config

    def generate(
        self,
        batch,
        **kwargs,
    ):
        self.set_routing_infos(batch, generate=True)

        generations = self.model.generate(
            inputs=batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs
        )
        return generations

    def on_save_checkpoint(self, ckpt):
        super().on_save_checkpoint(ckpt)

        # inject expert info in the expert checkpoint
        expert_info = ExpertInfo(
            expert_name=self.hparams.expert_name,
            expert_task_name=self.hparams.finetune_task_name,
            expert_config=self.modifier_config,
            training_config=self.training_config,
        )
        ckpt["expert_info"] = expert_info.asdict()

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        return super().on_before_optimizer_step(optimizer)

    def as_expert(self):
        state_dict = self.state_dict()
        self._delete_non_trainable_params(state_dict)

        # to use as an expert, we need to remove a `model.` prefix
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

        # inject expert info in the expert checkpoint
        expert_info = ExpertInfo(
            expert_name=self.hparams.expert_name,
            expert_task_name=self.hparams.finetune_task_name,
            expert_config=self.modifier_config,
            training_config=self.training_config,
        )
        return Expert(
            expert_info=expert_info,
            expert_weights=state_dict,
        )


class MultiExpertModel(ExpertModel):
    """Adds all functions and properties for a multi-expert model."""

    def __init__(self, **config_kwargs):
        config_kwargs["model_modifier"] = None
        super().__init__(**config_kwargs)

        self.experts_names = []

    @property
    def lock(self):
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
        return self._lock

    @property
    def experts_containers(self) -> List[ExpertContainer]:
        containers = []
        for _, module in self.model.named_modules():
            for _, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    containers.append(child)
        return containers

    @property
    def selectors(self) -> Dict[str, Selector]:
        return self.model.selectors

    def delete_expert_container(self):
        """
        Replaces the expert container with the expert with the given name.
        """
        for _, module in self.model.named_modules():
            for c_name, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    setattr(module, c_name, child.layer)
        self.experts_names.clear()

    def add_experts_from_library(self, library):
        import concurrent.futures

        import tqdm

        def add_module(self, module_name):
            expert_dump = library[module_name]
            self.add_expert_instance(expert_dump)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Create a list to hold the futures
            futures = []
            for element in library.keys():
                futures.append(executor.submit(partial(add_module, self), element))

            # Progress bar setup
            with tqdm.tqdm(
                total=len(library), desc="Adding experts...", unit="expert"
            ) as progress_bar:
                for _ in concurrent.futures.as_completed(futures):
                    progress_bar.update(1)

    def load_from_module_dict(self, module_dict, action="route"):
        for module_name, destination in module_dict.items():
            if isinstance(destination, str):
                self.load_expert(
                    destination,
                    module_name,
                    action=action,
                    is_default=module_name == "default",
                )
            elif isinstance(destination, Expert):
                self.add_expert_instance(
                    destination,
                    module_name,
                    action=action,
                    is_default=module_name == "default",
                )

    def add_empty_expert(
        self,
        expert_name,
        expert_config=None,
    ) -> Expert:
        """Adds a new empty expert to the model."""
        new_expert = Expert(
            expert_info=ExpertInfo(
                expert_name,
                expert_config=expert_config,
                expert_model=self.hparams.model,
                training_config=self.training_config,
            ),
        )

        new_expert = self.add_expert_instance(new_expert)
        logger.info("Added empty expert: {}".format(expert_name))
        return new_expert

    def load_expert(
        self,
        expert_path: str,
        expert_name: str = None,
        action: str = "merge",
        is_default: bool = False,
        expert_library: ExpertLibrary = None,
    ):
        from mttl.models.library.expert import load_expert

        expert = load_expert(
            expert_path,
            expert_name=expert_name,
            expert_library=expert_library,
        )

        if self.hparams.model != expert.training_config.model:
            raise ValueError(
                "The expert has been trained on top of a different model!"
                " Detected: {} - Expected: {}".format(
                    expert.training_config.model, self.hparams.model
                )
            )

        logger.info(
            f"Adding expert with name {expert.name}... with action ... {action}!"
        )
        self.add_expert_instance(expert, action=action, is_default=is_default)

    def add_expert_instance(
        self,
        expert_instance: Expert,
        expert_name=None,
        action="route",
        is_default=False,
    ) -> Expert:
        """
        If action is merge, then the expert is merged with the existing model, and we return None.
        """
        if expert_name is not None:
            # we want to load expert instance with a given name (might be different from the one in the expert instance)
            # we dont want to change expert instance though!
            # will create a copy for now (maybe safer), alternatively can change the name and set it back at the end of the function
            expert_instance = expert_instance.clone()
            expert_instance.name = expert_name

        with self.lock:
            add_expert_to_transformer(
                self.model,
                expert_instance,
                action=action,
                is_default=expert_instance.name == "default" or is_default,
                routing_config=self.selector_config,
                training_config=self.training_config,
            )

            if action != "merge":
                self.experts_names.append(expert_instance.name)
                # reload the expert instance to fill the weights properly if this was an empty expert
                expert_instance = self.get_expert_instance(expert_instance.name)
            return expert_instance

    def set_selector(
        self,
        modifier_type: str,
        selector_config: SelectorConfig,
        selector_weights: dict = None,
    ):
        from mttl.models.containers import replace_selector_for_container

        n_selectors, n_selectors_views = replace_selector_for_container(
            self.model,
            modifier_type,
            selector_config,
            selector_weights,
            force_replace=True,
        )
        assert self.model.selectors[modifier_type]
        logger.info(
            "Created {} selectors and {} views.".format(n_selectors, n_selectors_views)
        )

    def extract_parameters(self, p_name_pattern=".*lora.*"):
        """
        Extracts task embeddings for parameters matching the given pattern.

        Args:
            p_name_pattern (str, optional): Regular expression pattern to match parameter names.
                Defaults to ".*lora.*".

        Returns:
            torch.Tensor: Concatenated tensor of task embeddings for the matched parameters.
        """
        para_list = []
        for name, param in self.model.named_parameters():
            if re.fullmatch(p_name_pattern, name):
                para_list.append(param.reshape(-1))
        return torch.cat(para_list)

    def get_task_embeddings(self):
        """
        Retrieves the task embeddings for the loaded experts.

        This method assumes that the names of the loaded experts correspond to the tasks they are made for.

        Returns:
        embeddings (dict): A dictionary containing the task embeddings for each expert.
                           The keys are the expert names and the values are the corresponding embeddings.
        """
        if len(self.experts_names) == 0:
            return self.extract_parameters()

        embeddings = {}
        for exp_name in self.experts_names:
            embeddings[exp_name] = (
                self.extract_parameters(p_name_pattern=rf".*{exp_name}\..*lora.*")
                .detach()
                .cpu()
            )
        return embeddings

    def get_expert_instance(self, expert_name):
        """
        Retrieves an instance of the specified expert from the model.

        Args:
            expert_name (str): The name of the expert to retrieve.
            silent (bool, optional): If True, suppresses the ValueError exception when the expert is not found.
                Defaults to True.

        Returns:
            expert: An instance of the specified expert.

        Raises:
            AssertionError: If the expert name is not found in the model, if no expert containers are found,
                or if the expert names are not unique.
            ValueError: If the expert is not found in the model and silent is False.
        """
        assert (
            expert_name in self.experts_names
        ), f"Expert {expert_name} not found in the model."
        assert (
            len(self.experts_containers) > 0
        ), "No expert containers found in the model."
        assert len(set(self.experts_names)) == len(
            self.experts_names
        ), "Expert names are not unique."

        expert_params = {}
        for container in self.experts_containers:
            if expert_name in container.expert_infos:
                expert_info = container.expert_infos[expert_name]
                expert_weights = container._get_expert_weights(expert_name)
                expert_weights = {
                    f"{container.layer_name}.{k}": v for k, v in expert_weights.items()
                }
                expert_params.update(expert_weights)

        retrieved_expert = Expert(expert_info=expert_info, expert_weights=expert_params)
        return retrieved_expert

    def get_merged_expert(
        self, modifier_type: str = "lora", with_global_names=True, **kwargs
    ) -> Expert:
        """
        Converts the current expert model into an instance of the Expert class by merging the experts in the containers using weights from the corresponding selectors.

        Args:
            with_global_names (bool, optional): Whether to include global names in the merged weights. Defaults to True.

        Returns:
            Expert: An instance of the Expert class.

        Raises:
            None

        Example:
            model = ExpertModel()
            expert = model.to_expert(weights={'expert1': 0.5, 'expert2': 0.5}, with_global_names=True)
        """
        from mttl.models.modifiers.modify_model import get_modifier_type

        expert_params = {}
        assert (
            modifier_type in self.selectors
        ), f"Modifier type {modifier_type} not in model."
        for container in self.experts_containers:
            if not get_modifier_type(container.config) == modifier_type:
                logger.info(
                    f"Skipping container {container.layer_name} with modifier type {get_modifier_type(container.config)}"
                )
                continue

            params = container.get_merged_params(
                with_global_names=with_global_names, **kwargs
            )
            expert_params.update(params)
            expert_config = container.config
        if len(expert_params) == 0:
            raise ValueError(
                "No experts to merge found. Make sure the 'modifier_type' is correct."
            )

        expert_info = ExpertInfo(
            expert_name=self.training_config.finetune_task_name,
            expert_task_name=self.training_config.finetune_task_name,
            training_config=self.training_config,
            expert_config=expert_config,
        )
        return Expert(expert_info=expert_info, expert_weights=expert_params)

    def set_routing_infos(self, batch, generate=False):
        self.model.info_container["routing_infos"] = RoutingInfo.from_batch(batch)


def calculate_DPO_loss(
    original_prefered_logprob,
    original_disprefered_logprob,
    ref_prefered_logprob,
    ref_disprefered_logprob,
    beta=0.5,
):
    """
    Calculate the DPO loss.
    original_prefered_logprob: the logprob of the prefered expert in the original model
    original_disprefered_logprob: the logprob of the disprefered expert in the original model
    ref_prefered_logprob: the logprob of the prefered expert in the reference model
    ref_disprefered_logprob: the logprob of the disprefered expert in the reference model
    """

    original_prefered_relative_logprob = (
        original_prefered_logprob - ref_prefered_logprob
    )
    disprefered_relative_logprob = (
        original_disprefered_logprob - ref_disprefered_logprob
    )

    reward_accuracies = (
        (original_prefered_relative_logprob > disprefered_relative_logprob)
        .float()
        .mean(dim=-1)
    )
    reward_margins = (
        original_prefered_relative_logprob - disprefered_relative_logprob
    ).mean(dim=-1)

    loss = -F.logsigmoid(
        beta * (original_prefered_relative_logprob - disprefered_relative_logprob)
    ).mean(dim=-1)

    return loss, reward_accuracies, reward_margins


def get_log_prob(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)


class ExpertModelSimPO(EfficientCheckpointModule):
    def __init__(self, preference_model, **kwargs):
        super().__init__(**kwargs)
        self.preference_model = preference_model
        self.trainable_param_names = kwargs.get("trainable_param_names", None)
        self.beta = kwargs.get("beta", 0.5)
        self.loss_type = kwargs.get("loss_type", "sigmoid")
        self.label_smoothing = kwargs.get("label_smoothing", 0.1)
        # log hyperparameters
        self.save_hyperparameters(kwargs)

    def simpo_loss(
        self, original_prefered_logprob, original_disprefered_logprob, gamma_beta_ratio
    ):
        """
        Compute the SIMPO loss.

        ref: https://github.com/princeton-nlp/SimPO/blob/main/scripts/simpo_trainer.py

        args: original_prefered_logps: log probabiliteis of the prefered expert in the original model
              original_disprefered_logps: log probabiliteis of the disprefered expert in the original model
        """

        pi_logratios = original_prefered_logprob - original_disprefered_logprob
        logits = pi_logratios - gamma_beta_ratio

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Loss type {self.loss_type} not supported. Choose from ['sigmoid', 'hinge']"
            )

        chosen_rewards = (
            self.beta * original_prefered_logprob.detach()
        )

        reject_rewards = (
            -self.beta
            * original_disprefered_logprob.detach()
        )

        return losses, chosen_rewards, reject_rewards

    def training_step(self, batch, _):
        prompt_prefered_ids = batch["prompt_prefered_ids"]
        prompt_disprefered_ids = batch["prompt_disprefered_ids"]

        prompt_prefered_mask = batch["prompt_prefered_mask"]
        prompt_disprefered_mask = batch["prompt_disprefered_mask"]

        # original model
        model_prefered_log_prob = get_log_prob(
            self.preference_model.model.forward(
                prompt_prefered_ids, attention_mask=prompt_prefered_mask
            ).logits,
            labels=prompt_prefered_ids,
        )

        model_disprefered_log_prob = get_log_prob(
            self.preference_model.model.forward(
                prompt_disprefered_ids, attention_mask=prompt_disprefered_mask
            ).logits,
            labels=prompt_disprefered_ids,
        )

        loss, chosen_rewards, rejected_rewards = self.simpo_loss(
            model_prefered_log_prob, model_disprefered_log_prob, gamma_beta_ratio=0.1
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/chosen_rewards", chosen_rewards, on_step=True, on_epoch=True)
        self.log(
            "train/rejected_rewards", rejected_rewards, on_step=True, on_epoch=True
        )

        return loss

    def validation_step(self, batch, _):
        prompt_prefered_ids = batch["prompt_prefered_ids"]
        prompt_disprefered_ids = batch["prompt_disprefered_ids"]

        prompt_prefered_mask = batch["prompt_prefered_mask"]
        prompt_disprefered_mask = batch["prompt_disprefered_mask"]

        # original model
        model_prefered_log_prob = get_log_prob(
            self.preference_model.model.forward(
                prompt_prefered_ids, attention_mask=prompt_prefered_mask
            ).logits,
            labels=prompt_prefered_ids,
        )

        model_disprefered_log_prob = get_log_prob(
            self.preference_model.model.forward(
                prompt_disprefered_ids, attention_mask=prompt_disprefered_mask
            ).logits,
            labels=prompt_disprefered_ids,
        )

        loss, chosen_rewards, rejected_rewards = self.simpo_loss(
            model_prefered_log_prob, model_disprefered_log_prob, gamma_beta_ratio=0.1
        )
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/chosen_rewards", chosen_rewards, on_step=True, on_epoch=True)
        self.log("val/rejected_rewards", rejected_rewards, on_step=True, on_epoch=True)

        return loss


class ExpertModelDPO(EfficientCheckpointModule):

    def __init__(self, preference_model, ref_expert_model, **kwargs):
        super().__init__(**kwargs)
        self.preference_model = preference_model
        self.ref_expert_model = ref_expert_model
        self.trainable_param_names = kwargs.get("trainable_param_names", None)
        # log hyperparameters
        self.save_hyperparameters(kwargs)

    def training_step(self, batch, _):

        prompt_prefered_ids = batch["prompt_prefered_ids"]
        prompt_disprefered_ids = batch["prompt_disprefered_ids"]

        prompt_prefered_mask = batch["prompt_prefered_mask"]
        prompt_disprefered_mask = batch["prompt_disprefered_mask"]

        # original model
        model_prefered_log_prob = get_log_prob(
            self.preference_model.model.forward(
                prompt_prefered_ids, attention_mask=prompt_prefered_mask
            ).logits,
            labels=prompt_prefered_ids,
        )

        model_disprefered_log_prob = get_log_prob(
            self.preference_model.model.forward(
                prompt_disprefered_ids, attention_mask=prompt_disprefered_mask
            ).logits,
            labels=prompt_disprefered_ids,
        )

        # reference model
        ref_prefered_log_prob = get_log_prob(
            self.ref_expert_model.model.forward(
                prompt_prefered_ids, attention_mask=prompt_prefered_mask
            ).logits,
            labels=prompt_prefered_ids,
        )

        ref_disprefered_log_prob = get_log_prob(
            self.ref_expert_model.model.forward(
                prompt_disprefered_ids, attention_mask=prompt_disprefered_mask
            ).logits,
            labels=prompt_disprefered_ids,
        )

        loss, reward_accuracies, reward_margins = calculate_DPO_loss(
            model_prefered_log_prob,
            model_disprefered_log_prob,
            ref_prefered_log_prob,
            ref_disprefered_log_prob,
            beta=0.1,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log(
            "train/reward_accuracies",
            reward_accuracies,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/reward_margins",
            reward_margins,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, _):
        prompt_prefered_ids = batch["prompt_prefered_ids"]
        prompt_disprefered_ids = batch["prompt_disprefered_ids"]

        prompt_prefered_mask = batch["prompt_prefered_mask"]
        prompt_disprefered_mask = batch["prompt_disprefered_mask"]

        # original model
        model_prefered_log_prob = get_log_prob(
            self.preference_model.model.forward(
                prompt_prefered_ids, attention_mask=prompt_prefered_mask
            ).logits,
            labels=prompt_prefered_ids,
        )

        model_disprefered_log_prob = get_log_prob(
            self.preference_model.model.forward(
                prompt_disprefered_ids, attention_mask=prompt_disprefered_mask
            ).logits,
            labels=prompt_disprefered_ids,
        )

        # reference model
        ref_prefered_log_prob = get_log_prob(
            self.ref_expert_model.model.forward(
                prompt_prefered_ids, attention_mask=prompt_prefered_mask
            ).logits,
            labels=prompt_prefered_ids,
        )

        ref_disprefered_log_prob = get_log_prob(
            self.ref_expert_model.model.forward(
                prompt_disprefered_ids, attention_mask=prompt_disprefered_mask
            ).logits,
            labels=prompt_disprefered_ids,
        )

        loss, reward_accuracies, reward_margins = calculate_DPO_loss(
            model_prefered_log_prob,
            model_disprefered_log_prob,
            ref_prefered_log_prob,
            ref_disprefered_log_prob,
            beta=0.1,
        )

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val/reward_accuracies",
            reward_accuracies,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/reward_margins",
            reward_margins,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, _):
        prompt_prefered_ids = batch["prompt_prefered_ids"]
        prompt_disprefered_ids = batch["prompt_disprefered_ids"]

        prompt_prefered_mask = batch["prompt_prefered_mask"]
        prompt_disprefered_mask = batch["prompt_disprefered_mask"]

        # original model
        model_prefered_log_prob = get_log_prob(
            self.preference_model.model.forward(
                prompt_prefered_ids, attention_mask=prompt_prefered_mask
            ).logits,
            labels=prompt_prefered_ids,
        )

        model_disprefered_log_prob = get_log_prob(
            self.preference_model.model.forward(
                prompt_disprefered_ids, attention_mask=prompt_disprefered_mask
            ).logits,
            labels=prompt_disprefered_ids,
        )

        # reference model
        ref_prefered_log_prob = get_log_prob(
            self.ref_expert_model.model.forward(
                prompt_prefered_ids, attention_mask=prompt_prefered_mask
            ).logits,
            labels=prompt_prefered_ids,
        )

        ref_disprefered_log_prob = get_log_prob(
            self.ref_expert_model.model.forward(
                prompt_disprefered_ids, attention_mask=prompt_disprefered_mask
            ).logits,
            labels=prompt_disprefered_ids,
        )

        loss, reward_accuracies, reward_margins = calculate_DPO_loss(
            model_prefered_log_prob,
            model_disprefered_log_prob,
            ref_prefered_log_prob,
            ref_disprefered_log_prob,
            beta=0.1,
        )
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "test/reward_accuracies",
            reward_accuracies,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/reward_margins",
            reward_margins,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


class MoEModel(MultiExpertModel):
    def __init__(self, expert_library: ExpertLibrary = None, **kwargs):
        kwargs["top_k"] = kwargs["moe_top_k"]
        kwargs["emb_dim"] = kwargs["moe_emb_dim"]
        kwargs["rkhs_dim"] = kwargs["moe_rkhs_dim"]
        init_from_scratch = kwargs.get("init_from_scratch", False)

        super().__init__(**kwargs)

        if not self.hparams.library_id and expert_library is None or init_from_scratch:
            for i in range(self.hparams.moe_num_experts):
                # Adding a Skilled LoRA with 1 skill.
                exp_config = SkilledLoRAConfig(
                    n_skills=1,
                    modify_layers=self.hparams.modify_layers,
                    modify_modules=self.hparams.modify_modules,
                    lora_alpha=self.hparams.lora_alpha,
                    lora_dropout=self.hparams.lora_dropout,
                    lora_rank=self.hparams.lora_rank,
                    lora_init_b_random=True,
                    n_splits=self.hparams.n_splits,
                    phi_2_align_heads=self.hparams.phi_2_align_heads,
                )
                self.add_empty_expert(f"e{i}", exp_config)
            self.moe_num_experts = kwargs["moe_num_experts"]
        else:
            if expert_library is None:
                expert_library = ExpertLibrary.get_expert_library(
                    self.hparams.library_id
                )
            for i, expert in enumerate(sorted(list(expert_library.keys()))):
                self.add_expert_instance(expert_library[expert], expert_name=expert)
            self.moe_num_experts = i + 1
            if isinstance(
                self.selector_config, (ArrowSelectorConfig, HiddenStateComputerConfig)
            ):
                from projects.modular_llm.eval_library import patch_prototypes

                patch_prototypes(self, expert_library, self.hparams)

    def training_step(self, batch, _):
        loss = super().training_step(batch, _)
        total_loss = loss.clone()

        if (
            "routing_gates" in self.model.info_container
            and self.model.info_container["routing_gates"]
        ):
            num = 0.0
            entropy_of_avg = 0.0
            entropy_of_route = 0.0

            for values in self.model.info_container["routing_gates"]:
                # compute MI loss
                values = values.to(torch.float32)
                values = values.view(-1, values.shape[-1])
                probs = torch.softmax(values, -1)
                entropy_of_avg += -(
                    probs.mean(0) * torch.log(probs.mean(0) + 1e-6)
                ).sum(-1)
                entropy_of_route += -(probs * torch.log(probs + 1e-6)).sum(-1).mean(0)
                num += 1.0

            entropy_of_avg = entropy_of_avg / num
            entropy_of_route = entropy_of_route / num
            mi_loss = -entropy_of_avg + entropy_of_route

            self.log(
                f"{self._log_pref}train/entropy_of_route",
                entropy_of_route,
                on_step=True,
                prog_bar=True,
            )
            self.log(
                f"{self._log_pref}train/entropy_of_avg",
                entropy_of_avg,
                on_step=True,
                prog_bar=True,
            )
            self.log(
                f"{self._log_pref}train/mi_loss",
                mi_loss,
                on_step=True,
                prog_bar=True,
            )

            if self.hparams.moe_ent_reg > 0.0:
                total_loss += self.hparams.moe_ent_reg * mi_loss

            elif self.hparams.moe_ent_free_bits > 0.0:
                normalized_entropy = entropy_of_route / math.log(self.moe_num_experts)
                total_loss += (
                    (1.0 - normalized_entropy) >= self.hparams.moe_ent_free_bits
                ) * -entropy_of_route

            self.model.info_container["routing_gates"].clear()

        self.log(f"{self._log_pref}train/loss", loss, on_step=True, prog_bar=True)
        self.log(
            f"{self._log_pref}train/total_loss", total_loss, on_step=True, prog_bar=True
        )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return total_loss
