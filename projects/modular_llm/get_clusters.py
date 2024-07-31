import argparse
import os
from datetime import datetime

import numpy as np
import torch
from pytorch_lightning import seed_everything
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mttl.logging import logger, setup_logging
from mttl.models.library.expert_library import DatasetLibrary
from mttl.models.ranker.adapter_ranker import AdapterRankerHelper
from mttl.utils import remote_login


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
    subset_dataset = Subset(dataset, subset_indices)

    train_dataloader = DataLoader(
        subset_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    all_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    return train_dataloader, all_dataloader, dataset


class EncoderModel:

    def __init__(self, encoding, model, device="cuda"):
        if encoding == "classifier":
            model = AdapterRankerHelper.get_ranker_instance(
                ranker_model="classifier",
                ranker_path=model,
                device=device,
            )
            model.encoder_fn = model.get_text_encode
            postprocess = lambda x: x.cpu().detach().numpy()
        elif encoding == "embedding":
            model = SentenceTransformer(model, device=device)
            model.encoder_fn = model.encode
            postprocess = lambda x: x
        else:
            raise ValueError(f"Invalid encoding: {encoding}")

        self.model = model.to(device)
        self.encoding = encoding
        self.postprocess = postprocess

    def get_text_encode(self, text):
        return self.postprocess(self.model.encoder_fn(text))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--encoding", type=str, default="embedding")
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/sentence-t5-xxl"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--cutoff", type=int, default=-1, help="Number of examples to use."
    )
    parser.add_argument("--subsample", type=float, default=0.2)
    parser.add_argument("--num_clusters", type=int, default=8)
    parser.add_argument(
        "--output_dir", type=str, default=os.getenv("OUTPUT_DIR", "./output")
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for dataloader"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    setup_logging(args.output_dir)
    logger.info("Args: {}".format(vars(args)))
    remote_login()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EncoderModel(args.encoding, args.model, device=device)
    train_dataloader, all_dataloader, all_dataset = get_dataset(args)

    with torch.no_grad():
        embedding_list = [
            model.get_text_encode(batch["source"])
            for batch in tqdm(
                train_dataloader, total=len(train_dataloader), desc="dataset"
            )
        ]

    all_embedding = np.concatenate(embedding_list, axis=0)
    logger.info(f"all_embedding shape: {all_embedding.shape}")

    kmeans = KMeans(
        n_clusters=args.num_clusters,
        init="k-means++",
        n_init=10,
        random_state=args.seed,
    ).fit(all_embedding)

    def add_cluster_id(example):
        embedding = model.get_text_encode(example["source"])
        example["cluster_id"] = [str(i) for i in kmeans.predict(embedding)]
        return example

    dataset = all_dataset.map(add_cluster_id, batched=True, batch_size=args.batch_size)

    dataset_name = (
        f"local://{args.output_dir}/clusters-"
        f"{args.num_clusters}-{datetime.now().isoformat()}"
    )

    logger.info(f"Pushing dataset to {dataset_name}")
    DatasetLibrary.push_dataset(dataset, dataset_name)


if __name__ == "__main__":
    main()
