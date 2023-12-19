import os
import sys
import copy
import torch
import wandb
import re
import numpy as np
import seaborn as sns
from dataclasses import replace
from functools import partial
from matplotlib import pyplot as plt
from huggingface_hub import login
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything
from huggingface_hub import create_repo, login, HfApi

from projects.wiki_experts.train_experts_main import get_datamodule
from projects.wiki_experts.src.evolution.utils import (
    get_loss,
    init_wandb_logger,
    TableLogger,
)

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from mttl.models.modifiers.expert_containers.expert_library import (
    remove_outdated_experts_from_library,
    get_best_expert_for_task,
    get_best_expert_for_score,
    LocalExpertLibrary,
    HFExpertLibrary,
    ExpertLibrary,
    Score,
)
from projects.wiki_experts.src.evolution.train_router import train_router as train
from projects.wiki_experts.src.evolution.evaluators import (
    Evaluator,
    prepare_evaluator,
    EvalCallback,
)


from mttl.models.modifiers.expert_containers.module_graph import Expert

from projects.wiki_experts.src.evolution.config import (
    EvolExpertConfig,
    increase_version,
)
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.expert_model import MultiExpertModel
from projects.wiki_experts.src.evolution.experiment_state import ExperimentState
from mttl.vllm_engines.engines import free_memory
from projects.wiki_experts.src.evolution.transfer_matrix import (
    eval_all_experts_on_task,
    eval_expert_on_task,
)
from mttl.datamodule.base import DefaultDataModule
from mttl.models.modifiers.expert_containers.library_transforms import (
    SVDEmbeddingTransform,
    SVDEmbeddingTransformConfig,
)


import os
import itertools
import pandas as pd
from huggingface_hub import login, HfApi, logout
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from mttl.models.modifiers.expert_containers.module_graph import load_expert

hf_api_key = "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"
login(token=hf_api_key)
# who am I
user = HfApi(token=hf_api_key).whoami()
print(user)
# set environment variable
os.environ["HF_TOKEN"] = hf_api_key

# hf_repo_id="oostapeno/flan-lib-neo-1B-20phi"
hf_repo_id = "sordonia/library-phi_2-v2"
# expert_lib =  HFExpertLibrary(hf_repo_id)
local_lib_location = f"/tmp/{hf_repo_id}"
os.makedirs(local_lib_location, exist_ok=True)
# local_lib_location = "/home/v-oostapenko/mttl_tmp/sordonia_phi2_lib"
# expert_lib:LocalExpertLibrary = LocalExpertLibrary.from_remote(expert_lib, local_lib_location)
expert_lib = LocalExpertLibrary(local_lib_location)

##########################################################################################


def create_embeddings():
    svd_embedder = SVDEmbeddingTransform(
        SVDEmbeddingTransformConfig(sparsity_threshold=0.5)
    )
    svd_embedder.transform(expert_lib, upload_to_hf=True)
    del svd_embedder


# module to embedding
embeddings = {}
for n, m in expert_lib.items():
    # emb = m.expert_weights.values()
    # embeddings[n] = [torch.flatten(e).cpu().numpy() for e in emb]
    # embeddings[n] = np.concatenate(embeddings[n])
    embeddings[n] = expert_lib.get_svd_embedding(n)
    if embeddings[n] is None:
        create_embeddings()
        embeddings[n] = expert_lib.get_svd_embedding(n)

# simmilarity table
n_tasks = len(expert_lib)
similarity_table = np.zeros((n_tasks, n_tasks))
for i, (n1, e1) in enumerate(embeddings.items()):
    if i == n_tasks:
        break
    for j, (n2, e2) in enumerate(embeddings.items()):
        if j == n_tasks:
            break
        similarity_table[i, j] = np.dot(e1, e2) / (
            np.linalg.norm(e1) * np.linalg.norm(e2)
        )
        # round
        similarity_table[i, j] = round(similarity_table[i, j], 4)


##########################################################################################
# create pirs of tasks and their similarity, then we order
pairs = {}
for i in range(similarity_table.shape[0]):
    for j in range(similarity_table.shape[1]):
        pair = (i, j)
        if i != j and (i, j) not in pairs and (j, i) not in pairs:
            pair = (i, j)
            pairs[pair] = similarity_table[i, j]
pairs = [(i, j, v) for (i, j), v in pairs.items()]
pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
# sample 20 points linraly over len(pairs)
tasks = list(embeddings.keys())[:n_tasks]
selected_pair_ids = list(np.linspace(0, len(pairs) - 1, 15, dtype=int))
selected_pairs = [pairs[i] for i in selected_pair_ids]

for p in selected_pairs:
    print(f"Pair: {tasks[p[0]]}, {tasks[p[1]]}, similarity: {p[2]}")


# tasks = list(embeddings.keys())[:n_tasks]
# distance = 1 - similarity_table
# _squareform = squareform(distance)
# linkage_matrix = linkage(_squareform, method='complete')
# cluster_dicts = []

# for num_clusters in range(5, 30, 5):
#     clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
#     task_indices = np.arange(len(similarity_table))
#     cluster_dict = {}
#     for task, cluster in zip(task_indices, clusters):
#         if cluster not in cluster_dict:
#             cluster_dict[cluster] = []
#         cluster_dict[cluster].append(task)
#         # Print or use the clusters
#     intracluster_distances = []
#     for cluster, idxs in cluster_dict.items():
#         print(f"Cluster {cluster}: {[tasks[i] for i in idxs]}")
#         intracluster_distances.append(np.mean(distance[idxs][:, idxs]))

#     cluster_dicts.append((cluster_dict, np.mean(intracluster_distances)))

# cluster_dict = sorted(cluster_dicts, key=lambda x: x[1])[0][0]
# for cluster, idxs in cluster_dict.items():
#     print(f"Cluster {cluster}: {[tasks[i] for i in idxs]}")


# # negative clusters
# tasks = list(embeddings.keys())[:n_tasks]
# distance = 1 - similarity_table
# _squareform = squareform(distance)
# linkage_matrix = linkage(_squareform, method='complete')
# num_clusters = 5
# clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
# task_indices = np.arange(len(similarity_table))
# cluster_dict = {}
# for task, cluster in zip(task_indices, clusters):
#     if cluster not in cluster_dict:
#         cluster_dict[cluster] = []
#     cluster_dict[cluster].append(task)
#     # Print or use the clusters
# for cluster, idxs in cluster_dict.items():
#     print(f"Negative cluster {cluster}: {tasks[idxs]}")
