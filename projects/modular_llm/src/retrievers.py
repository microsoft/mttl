import copy

import numpy as np
import torch

from mttl.config import ExpertConfig
from mttl.logging import logger
from mttl.models.expert_model import MultiExpertModel
from mttl.models.library.expert import Expert
from mttl.models.library.expert_library import VirtualLocalLibrary
from mttl.models.library.library_transforms import LibraryTransform
from mttl.models.library.utils import get_svd_embedding

RETRIEVERS = {}


def register_retriever(name):
    def decorator(cls):
        if name in RETRIEVERS:
            raise ValueError(f"Retriever {name} already registered")
        RETRIEVERS[name] = cls
        return cls

    return decorator


class Retriever(LibraryTransform):
    def __init__(self, config: ExpertConfig, sk=None, retriever_include_parent=True):
        super().__init__(config)
        self.sk = sk if sk is not None else config.sk
        self.retriever_include_parent = retriever_include_parent

    def prepare_transform(self, expert_lib):
        expert_lib_copy = VirtualLocalLibrary.from_expert_library(
            expert_lib, expert_lib.repo_id
        )
        return expert_lib_copy

    def finalize_transform(self, expert_lib, task, task_expert, sel_exp_names):
        # create the resulting library
        resulting_library = VirtualLocalLibrary.from_expert_library(
            expert_lib, expert_lib.repo_id
        )
        if task_expert is not None and self.retriever_include_parent:
            sel_exp_names[-1] = task_expert.name

        for n, metadata in list(resulting_library.data.items()):
            if n not in sel_exp_names:
                resulting_library.remove_expert(n)
        return resulting_library

    def transform(self, **kwargs):
        raise NotImplementedError()


@register_retriever("random")
class RandomRetriever(Retriever):
    def transform(
        self, expert_lib, current_task, task_expert: Expert = None, **kwargs
    ) -> VirtualLocalLibrary:
        expert_lib_copy = self.prepare_transform(expert_lib)
        if self.sk <= 0 or self.sk >= len(expert_lib_copy):
            return expert_lib_copy

        logger.disabled = True

        if task_expert is None:
            keys = list(set(expert_lib_copy.keys()))
        else:
            keys = list(set(expert_lib_copy.keys()) - {task_expert.name})

        sel_exp_names = np.random.choice(keys, self.sk, replace=False)
        resulting_library = self.finalize_transform(
            expert_lib, current_task, task_expert, sel_exp_names
        )

        logger.info(
            "Retrieved experts: {} with metric {}".format(
                list(resulting_library.keys()), "random"
            )
        )
        logger.disabled = False
        return resulting_library


def get_lora_task_embeddings(module: MultiExpertModel):
    """
    Retrieves the task embeddings for the loaded experts.

    This method assumes that the names of the loaded experts correspond to the tasks they are made for.

    Returns:
    embeddings (dict): A dictionary containing the task embeddings for each expert.
                        The keys are the expert names and the values are the corresponding embeddings.
    """
    if len(module.experts_names) == 0:
        return module.extract_parameters()

    embeddings = {}
    for exp_name in module.experts_names:
        embeddings[exp_name] = (
            module.extract_parameters(p_name_pattern=rf".*{exp_name}\..*lora.*")
            .detach()
            .cpu()
        )
    return embeddings


@register_retriever("lora_sim")
class LoraSimRetriever(Retriever):
    def transform(
        self,
        expert_lib,
        current_task,
        task_expert: Expert,
        module: MultiExpertModel,
        **kwargs,
    ) -> VirtualLocalLibrary:
        expert_lib_copy = self.prepare_transform(expert_lib)
        if self.sk <= 0 or self.sk >= len(expert_lib_copy):
            return expert_lib_copy

        assert task_expert is not None
        assert task_expert in expert_lib_copy

        module = copy.deepcopy(module)
        module.add_experts_from_library(expert_lib_copy)

        from torch.nn.functional import cosine_similarity

        task_module_name = task_expert.name

        # compute cosine similarity between each expert and current task's expert, keep top sk
        emb_tasks = module.get_lora_task_embeddings()

        # compare this task's embed with  other
        if task_module_name not in emb_tasks:
            return expert_lib_copy
        task_emb = emb_tasks[task_module_name]
        similarities = []
        t_names = []
        for t, emb in emb_tasks.items():
            if t != task_module_name:
                similarities.append(
                    cosine_similarity(task_emb.unsqueeze(0), emb.unsqueeze(0)).item()
                )
                t_names.append(t)
        similarities = {k: v for k, v in zip(t_names, similarities)}
        sel_exp_names = sorted(similarities, key=similarities.get, reverse=True)[
            : self.sk
        ]

        resulting_library = self.finalize_transform(
            expert_lib, current_task, task_expert, sel_exp_names
        )

        logger.info(
            "Retrieved experts: {} with metric {}".format(
                list(resulting_library.keys()), "LoraSimilarity"
            )
        )
        logger.disabled = False
        return resulting_library


@register_retriever("svdemb")
class SVDEmbeddingRetriever(Retriever):
    def transform(
        self,
        expert_lib,
        current_task,
        task_expert: Expert,
        **kwargs,
    ) -> VirtualLocalLibrary:
        expert_lib_copy = self.prepare_transform(expert_lib)
        if self.sk <= 0 or self.sk >= len(expert_lib_copy):
            return expert_lib_copy

        assert task_expert is not None
        assert task_expert in expert_lib_copy
        from torch.nn.functional import cosine_similarity

        task_module_name = task_expert.name
        # compute cosine similarity between each expert and current task's expert, keep top sk
        emb_tasks = {}
        # TODO: fix this when all pieces are merged + add test for retrievers
        emb_tasks[task_module_name] = get_svd_embedding(
            expert_lib_copy, task_module_name
        )
        for key, metadatum in expert_lib_copy.data.items():
            emb_tasks[key] = get_svd_embedding(expert_lib_copy, metadatum.expert_name)
            emb_tasks[key] = torch.tensor(emb_tasks[metadatum.expert_name])

        # compare this task's embed with  other
        task_emb = emb_tasks[task_module_name]
        similarities = []
        t_names = []
        for t, emb in emb_tasks.items():
            if t != task_module_name:
                similarities.append(
                    cosine_similarity(task_emb.unsqueeze(0), emb.unsqueeze(0)).item()
                )
                t_names.append(t)
        similarities = {k: v for k, v in zip(t_names, similarities)}
        sel_exp_names = sorted(similarities, key=similarities.get, reverse=True)[
            : self.sk
        ]

        resulting_library = self.finalize_transform(
            expert_lib, current_task, task_expert, sel_exp_names
        )

        logger.info(
            "Retrieved experts: {} with metric {}".format(
                list(resulting_library.keys()), "SVDEmbeddingRetriever"
            )
        )
        logger.disabled = False
        return resulting_library
