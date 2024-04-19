from functools import partial
import math
import re
import threading
from typing import Dict, List, Union
import numpy as np
import torch
import tqdm
from transformers import PreTrainedModel

from mttl.models.modifiers.expert_containers.library_transforms import (
    ArrowConfig,
    HiddenStateComputerConfig,
)
from mttl.models.modifiers.lora import SkilledLoRAConfig

from mttl.models.modifiers.expert_containers import add_expert_to_transformer
from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.models.modifiers.routing import RoutingInfo
from mttl.utils import logger
from mttl.models.modifiers.expert_containers.expert import Expert, ExpertInfo
from mttl.models.modifiers.expert_containers.expert_containers import (
    ExpertContainer,
    LoRAExpertContainer,
)
from mttl.models.modifiers.expert_containers.selectors import Selector, SelectorConfig


import torch
from collections import defaultdict
from torch.optim.optimizer import Optimizer

from mttl.models.llama_patch import replace_attn_with_flash_attn
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.base import ModifierConfig

from mttl.models.modifiers.expert_containers.expert import ExpertInfo
from mttl.models.modifiers.expert_containers.selectors import SelectorConfig
from mttl.models.modifiers.routing import RoutingInfo
from mttl.models.utils import (
    EfficientCheckpointModule,
    prepare_model_for_kbit_training,
)
from mttl.models.expert_config import ExpertConfig
from mttl.models.ranker.adapter_ranker import AdapterRankerHelper


torch.set_float32_matmul_precision("high")


class ArgmaxWeightedLoss(torch.nn.Module):
    def forward(self, logits, labels) -> float:
        pass


class XEntLoss(torch.nn.Module):
    def forward(self, logits, labels) -> float:
        pass


class UnlikelihoodLoss(torch.nn.Module):
    def forward(self, logits, labels) -> float:
        pass


class ExpertModel(EfficientCheckpointModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = kwargs.pop("tokenizer", None)
        model_object = kwargs.pop("model_object", None)

        # log hyperparameters
        self.save_hyperparameters(kwargs)

        self.load_in_8bit = kwargs.get("load_in_8bit", False)
        self.model: PreTrainedModel = None
        self.accumulate_metrics_batch = defaultdict(list)

        if model_object is None:
            from mttl.models.utils import model_loader_helper

            model_object = model_loader_helper(
                self.hparams.model,
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
        self._inference_outputs += [(loss.detach().cpu(),)]
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
        self._inference_outputs += [(loss.detach().cpu(),)]
        return mean_loss

    def log_loss(self, split="val"):
        outputs = self._inference_outputs
        losses = torch.cat([out[0] for out in outputs], 0)
        self._inference_outputs.clear()
        self.log(
            f"{self._log_pref}{split}/loss", losses.mean(), on_epoch=True, prog_bar=True
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
        import tqdm
        import concurrent.futures

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
                total=len(library), desc="Processing", unit="module"
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
        from mttl.models.modifiers.expert_containers.expert import load_expert

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

    def load_from_library(self, library, subsample_library_experts=0):
        keys = list(library.keys())
        if self.hparams.subsample_library_experts > 0:
            keys = np.random.permutation(keys)[:subsample_library_experts]

        for expert_name in tqdm.tqdm(keys, desc="Loading experts..."):
            expert_dump = library.get_expert(expert_name, with_auxiliary_data=True)
            self.add_expert_instance(expert_dump)

    def set_selector(
        self,
        modifier_type: str,
        selector_config: SelectorConfig,
        selector_weights: dict = None,
    ):
        from mttl.models.modifiers.expert_containers import (
            replace_selector_for_container,
        )

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
                # can we break here? or do we need to check all containers?
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
        ), f"Modifie type {modifier_type} not in model."
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


class MoEModel(MultiExpertModel):
    def __init__(self, **kwargs):
        kwargs["top_k"] = kwargs["moe_top_k"]
        kwargs["emb_dim"] = kwargs["moe_emb_dim"]
        kwargs["rkhs_dim"] = kwargs["moe_rkhs_dim"]
        library = kwargs.pop("expert_library", None)
        super().__init__(**kwargs)

        if not self.hparams.library_id and library is None:
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
            if library is None:
                library = ExpertLibrary.get_expert_library(self.hparams.library_id)
            for i, expert in enumerate(sorted(list(library.keys()))):
                self.add_expert_instance(library[expert], expert_name=f"e{i}")

            self.moe_num_experts = i + 1
            if isinstance(
                self.selector_config, (ArrowConfig, HiddenStateComputerConfig)
            ):
                from projects.wiki_experts.eval_library import patch_prototypes

                patch_prototypes(self, library, self.selector_config)

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
