from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from mttl.models.ranker.adapter_ranker import AdapterRankerHelper
from tqdm import tqdm
from nomic import atlas
import numpy as np
from sklearn.cluster import KMeans
import argparse
import logging
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from sentence_transformers import SentenceTransformer
from datasets import concatenate_datasets
import os
import huggingface_hub

huggingface_token = os.environ.get("HF_TOKEN")
huggingface_hub.login(token=huggingface_token)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)

parser.add_argument("--subsample", type=float, default=0.2)

parser.add_argument("--num_clusters", type=int, default=256)

parser.add_argument("--dataset", type=str, default="orca")

parser.add_argument("--output_file", type=str, default="orca_cluster.json")

parser.add_argument("--encoding", type=str, default="classifier")

parser.add_argument(
    "--new_dataname", type=str, default="zhan1993/flan-10k-flat-cluster-embedding"
)

parser.add_argument(
    "--model", type=str, default="sentence-transformers/sentence-t5-xxl"
)
args = parser.parse_args()
np.random.seed(42)


def get_text_encode(text, model):

    if args.encoding == "classifier":
        return model.get_text_encode(text).cpu().detach().numpy()
    elif args.encoding == "embedding":
        return model.encode(text)


if args.encoding == "classifier":
    model = AdapterRankerHelper.get_ranker_instance(
        ranker_model="classifier",
        ranker_path="zhan1993/classifier_ranker_t5_v3",
    )
elif args.encoding == "embedding":
    model = SentenceTransformer(args.model)

def get_dataset(args):
    dataset = DatasetLibrary.pull_dataset(args.dataset, split="train")

    # create the subsample of the dataset if cutoff is set.
    if args.cutoff > 0:
        dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.select(range(args.cutoff))

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(args.subsample * dataset_size))
    subset_indices = indices[:split]
    subset_dataset = dataset.select(subset_indices)

    train_dataloader = DataLoader(
        subset_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    all_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    return train_dataloader, all_dataloader, dataset, subset_dataset

# def get_orca_dataset():

#     dataset = load_dataset("Open-Orca/OpenOrca")

#     # create the subsample of the dataset.
#     dataset_size = len(dataset["train"])
#     indices = list(range(dataset_size))
#     # random indices
#     np.random.shuffle(indices)
#     split = int(np.floor(args.subsample * dataset_size))
#     subset_indices = indices[:split]
#     subset_dataset = Subset(dataset["train"], subset_indices)

#     train_dataloader = DataLoader(
#         subset_dataset, batch_size=args.batch_size, num_workers=8
#     )
#     all_dataloader = DataLoader(
#         dataset["train"], batch_size=args.batch_size, num_workers=8
#     )

#     return train_dataloader, all_dataloader, dataset["train"]


# def get_flan_dataset():

#     flan = FlanModule(
#         FlanConfig(
#             model="EleutherAI/gpt-neo-125m",
#             model_family="gpt",
#             train_batch_size=4,
#             predict_batch_size=4,
#             dataset="sordonia/flan-10k-flat",
#             remove_phi_eval_tasks=True,
#         )
#     )

#     dataset = flan.train_dataset
#     # create the subsample of the dataset.
#     dataset_size = len(dataset)
#     indices = list(range(dataset_size))
#     # random indices
#     np.random.shuffle(indices)
#     split = int(np.floor(args.subsample * dataset_size))
#     subset_indices = indices[:split]
#     subset_dataset = Subset(dataset, subset_indices)

#     train_dataloader = DataLoader(
#         subset_dataset, batch_size=args.batch_size, num_workers=8
#     )
#     all_dataloader = flan.train_dataloader()

#     all_dataset = concatenate_datasets(
#         [flan.train_dataset, flan.dev_dataset, flan.test_dataset]
#     )

#     return train_dataloader, all_dataloader, all_dataset


if __name__ == "__main__":
    
    train_dataloader, all_dataloader, all_dataset, subset_dataset = get_dataset(args)
    breakpoint()

    embedding_list = []

    for i, batch in tqdm(
        enumerate(train_dataloader), total=len(train_dataloader), desc="dataset"
    ):
        if "source" in batch:
            embedding = get_text_encode(batch["source"], model)
        else:
            embedding = get_text_encode(batch["question"], model)
        embedding_list.append(embedding)

    all_embedding = np.concatenate(embedding_list, axis=0).reshape(-1, 768)
    logger.info(f"all_embedding shape: {all_embedding.shape}")
    kmeans = KMeans(
        n_clusters=args.num_clusters,
        init="k-means++",
        n_init=10,
        random_state=42,
    ).fit(all_embedding)

    # map the new item with kmeans cluster

    def add_cluster_id(example):
        if "source" in example:
            embedding = get_text_encode(example["source"], model)
        else:
            embedding = get_text_encode(example["question"], model)
        embedding = embedding.reshape(1, -1)
        example["cluster_id"] = str(kmeans.predict(embedding)[0])
        return example

    # all_dataset = all_dataset.select(list(range(100)))

    dataset = all_dataset.map(add_cluster_id)

    # Push the merged dataset back to Hugging Face Hub
    dataset.push_to_hub(args.new_dataname)
