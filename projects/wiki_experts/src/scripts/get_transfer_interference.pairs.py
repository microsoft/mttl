import os
import numpy as np
import seaborn as sns
from dataclasses import replace
from functools import partial
from matplotlib import pyplot as plt
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything
from mttl.models.modifiers.expert_containers.expert_library import (
    LocalExpertLibrary,
    get_expert_library,
)
from mttl.models.modifiers.expert_containers.library_transforms import (
    SVDEmbeddingTransform,
    SVDEmbeddingTransformConfig,
)
from mttl.utils import remote_login
from projects.wiki_experts.src.evolution.utils import get_svd_embedding

remote_login()

# hf_repo_id="oostapeno/flan-lib-neo-1B-20phi"
# hf_repo_id = "ostapeno/library-gptneo_1B_flan_2ep"
hf_repo_id = "ostapeno/library-stablelm_flan_5ep"

local_lib_location = f"/tmp/{hf_repo_id}"
if os.path.exists(local_lib_location):
    expert_lib = LocalExpertLibrary(local_lib_location)
    expert_lib.update_from_expert_library(hf_repo_id)
else:
    expert_lib = get_expert_library(hf_repo_id)
    os.makedirs(local_lib_location, exist_ok=True)
    expert_lib: LocalExpertLibrary = LocalExpertLibrary.from_expert_library(
        expert_lib, local_lib_location
    )
##########################################################################################

sparsity_threshold = 0.1
if "neo" in hf_repo_id:
    sparsity_threshold = 0.7
elif "phi" in hf_repo_id:
    sparsity_threshold = 0.5


def create_embeddings():
    svd_embedder = SVDEmbeddingTransform(
        SVDEmbeddingTransformConfig(sparsity_threshold=sparsity_threshold),
        random_state=42,
    )
    svd_embedder.transform(expert_lib, persist=True, force=True)
    del svd_embedder


# embeds = expert_lib.get_auxiliary_data("embeddings")
# if len(embeds) == 0:
print("creating embeddings")
create_embeddings()

# module to embedding
embeddings = {}
for n, m in expert_lib.items():
    embeddings[n] = get_svd_embedding(expert_lib, n)

# assert that no NaNs in embeddings
for n, e in embeddings.items():
    assert not np.isnan(e).any(), n

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
# plots is a list of pairs of tasks and their similarity
n_pairs = 50
tasks = list(embeddings.keys())
# Extract the third element from each tuple
similarities = [item[2] for item in pairs]

# Find the range of the third element
min_value = min(similarities)
max_value = max(similarities)

# Create bins based on the range of the third element
bins = np.linspace(min_value, max_value, n_pairs + 1)
bin_indices = np.digitize(similarities, bins)

sampled_indices = [
    np.random.choice(np.where(bin_indices == i)[0]) for i in range(1, n_pairs + 1)
]
selected_pairs = [pairs[i] for i in sampled_indices]

for p in selected_pairs:
    print(f"Pair: {tasks[p[0]]}, {tasks[p[1]]}, similarity: {p[2]}")
