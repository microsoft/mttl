import os
import sys
import glob
import torch
import faiss
import pickle
import numpy as np
import glob as glob
from os.path import join
from pytorch_lightning import seed_everything

from mttl.config import parse_config
from mttl.cluster_tuning.encodings import Encodings, ClusterInfos


def main(config):
    """
    Get cluster assignments
    """
    use_pca = -1
    use_normalization = True
    use_centering = False
    use_constraints = False
    algo = "kmeans_balanced"

    subsample = 500_000
    clusters = [5, 10, 15, 20, 25, 30, 35, 50]

    chunks = glob.glob(f'{config.embeddings_path}*-chunk*')
    if not len(chunks):
        # we still expand to help a bit the user
        chunks = [list(glob.glob(config.embeddings_path))[0]]

    # load all chunks and concatenate to the above lists
    data = None

    train_chunks = len(chunks) - 1 or 1
    subsample_chunk = subsample // train_chunks

    for i, chunk in enumerate(chunks):
        print(f'loading {i}/{len(chunks)} chunk', end='\r')
        chunk_data = Encodings.load(chunk)

        # subsample by num of chunks
        indices = np.random.choice(
            len(chunk_data.hashes), min(subsample_chunk, len(chunk_data.hashes)), replace=False
        )
        if data is None:
            data = Encodings(input_type=chunk_data.input_type)
        # the input type should be consistent across all chunks
        assert data.input_type == chunk_data.input_type
        # pick indices from hashes, embeds, task_ids and flags
        data.hashes.extend(np.asarray(chunk_data.hashes)[indices].tolist())
        data.encodings.extend(np.asarray(chunk_data.encodings)[indices].tolist())
        data.task_ids.extend(np.asarray(chunk_data.task_ids)[indices].tolist())
        data.is_test.extend(np.asarray(chunk_data.is_test)[indices].tolist())

        if i == 1 : break

    # name of files
    def filename(n_clusters):
        return f"{n_clusters}"

    cluster_dir = config.output_dir

    data.encodings = np.asarray(data.encodings).astype("float32")
    data.is_test = np.asarray(data.is_test)
    training_embeds = data.encodings[(data.is_test == 0)]

    if use_normalization:
        training_embeds = training_embeds / np.linalg.norm(
            training_embeds, axis=1, keepdims=True
        )

    if use_centering:
        torch_embeds = torch.from_numpy(training_embeds).T
        s, v, d = torch.linalg.svd(torch_embeds)
        centering = torch.matmul(s[:, :1], s[:, :1].T)
        torch_embeds = torch_embeds - centering.mm(torch_embeds)
        training_embeds = torch_embeds.T.numpy().astype("float32")

    if use_pca > 0:
        mat = faiss.PCAMatrix(training_embeds.shape[1], use_pca)
        mat.train(training_embeds)
        training_embeds = mat.apply(training_embeds)

    for n_clusters in clusters:
        kmeans = None
        print(f"{n_clusters} clusters")

        example_to_ids_path = join(cluster_dir, f"{filename(n_clusters)}.pth")

        # check if the centroids have been computed without soft clustering
        print("computing cluster assignments")

        if 'kmeans' in algo:
            if 'faiss' in algo:
                kmeans = faiss.Kmeans(
                    training_embeds.shape[-1],
                    int(n_clusters),
                    niter=30,
                    verbose=True,
                    gpu=True,
                    nredo=2,
                    max_points_per_centroid=10_000_000,
                )
                kmeans.train(training_embeds)
                index = kmeans.index
            elif 'balanced' in algo:
                # https://github.com/pclucas14/balanced-kmeans
                # sys.path += [join(os.environ.get("HOME"), 'balanced-kmeans')]
                from kmeans_pytorch import KMeans
                device = torch.device('cuda')
                kmeans = KMeans(
                    n_clusters=n_clusters, 
                    device=device,
                    balanced=True 
                )
                kmeans.fit(
                    torch.from_numpy(training_embeds).cuda(), 
                    distance='euclidean', 
                    iter_limit=30, 
                )
                centroids = kmeans.cluster_centers
                index = faiss.IndexFlatL2(centroids.size(-1))
                index.add(centroids.cpu())

            cluster_infos = ClusterInfos(input_type=data.input_type)

            # assignment
            for chunk in chunks:
                chunk_data = Encodings.load(chunk)
                embeds = np.asarray(chunk_data.encodings).astype("float32")

                if use_normalization:
                    embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

                if use_centering:
                    embeds = torch.from_numpy(embeds).T
                    embeds = embeds - centering.mm(embeds)
                    embeds = embeds.T.numpy().astype("float32")

                if use_pca > 0:
                    embeds = mat.apply(embeds)

                D, I = index.search(embeds, int(n_clusters))

                cluster_infos.hashes.extend(chunk_data.hashes)
                cluster_infos.task_names.extend(chunk_data.task_names)
                cluster_infos.cluster_ids.extend(torch.from_numpy(I[:, 0]).flatten().tolist())
                cluster_infos.is_test.extend(chunk_data.is_test)
                assert chunk_data.input_type == cluster_infos.input_type

                distances = np.zeros((D.shape[0], D.shape[1]))
                for i in range(D.shape[0]):
                    for j in range(D.shape[1]):
                        distances[i, I[i, j]] = D[i, j]
                cluster_infos.cluster_dists.extend(distances.tolist())

        assert len(cluster_infos.hashes) == len(cluster_infos.cluster_ids)
        cluster_sizes = np.bincount(cluster_infos.cluster_ids)

        print("Sorted cluster sizes:", sorted(cluster_sizes))
        print(
            "Bigger to smaller ratio:",
            np.max(cluster_sizes) / (np.min(cluster_sizes) + 0.1),
        )
        cluster_infos.save(example_to_ids_path)


if __name__ == "__main__":
    config = parse_config()

    # Setup config
    seed_everything(config.seed)

    # ...
    main(config)
