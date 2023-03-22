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


def balance_clusters(indices, distances, centroids, reprs):
    # distances
    dists = np.zeros((distances.shape[0], distances.shape[1]))
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            dists[i, indices[i, j]] = distances[i, j]

    size_of_clusters = np.bincount(indices[:, 0])
    sorted_clusters = np.argsort(size_of_clusters)[::-1]

    examples_by_cluster = [set() for _ in range(len(sorted_clusters))]
    for example in range(len(indices)):
        examples_by_cluster[indices[example, 0]].add(example)

    K = len(indices) // len(sorted_clusters)

    for i, donor_cluster in enumerate(sorted_clusters[:-1]):
        # distances to all other clusters
        donor_examples = list(examples_by_cluster[donor_cluster])
        # Examples x Clusters
        donor_dists = dists[np.asarray(donor_examples)]
        # sort by minimal distance to other clusters
        donor_dists[:, sorted_clusters[: i + 1]] = np.inf
        # the min distance to any other cluster than the donor cluster
        # (Examples,): Distance to the closest cluster
        closest_clusters_dist = np.min(donor_dists, axis=-1)
        # (Examples,): Id of the closest cluster
        closest_clusters_idx = np.argmin(donor_dists, axis=-1)
        # (Examples,): Closest examples to any cluster
        sorted_closest_examples_idx = np.argsort(closest_clusters_dist)
        # (Examples,): Examples sorted by closeness to any cluster

        sorted_closest_examples = [
            donor_examples[i] for i in sorted_closest_examples_idx
        ]
        sorted_closest_cluster = [
            closest_clusters_idx[i] for i in sorted_closest_examples_idx
        ]

        for cand_ex, receiver_cluster in zip(
            sorted_closest_examples, sorted_closest_cluster
        ):
            if len(examples_by_cluster[donor_cluster]) <= K:
                break

            # transfer the example to the receiver cluster
            examples_by_cluster[receiver_cluster].add(cand_ex)
            examples_by_cluster[donor_cluster].remove(cand_ex)

    # recompute centroids now
    centroids = np.zeros(centroids.shape)
    for cluster in range(len(sorted_clusters)):
        centroids[cluster, :] = reprs[
            np.asarray(list(examples_by_cluster[cluster]))
        ].mean(0)
    return centroids


def main(config):
    """
    Get cluster assignments
    """

    use_pca = -1
    use_normalization = False
    use_centering = False
    use_constraints = False
    algo = "faiss"

    subsample = 500_000
    clusters = [1, 5, 10, 15, 20, 25, 30, 35, 50]

    chunks = glob.glob(config.embeddings_path + "-chunk*")
    if not len(chunks):
        # we still expand to help a bit the user
        chunks = [list(glob.glob(config.embeddings_path))[0]]

    # load all chunks and concatenate to the above lists
    data = Encodings()

    train_chunks = len(chunks) - 1 or 1
    subsample_chunk = subsample // train_chunks

    for chunk in chunks:
        chunk_data = Encodings.load(chunk)
        # subsample by num of chunks
        indices = np.random.choice(
            len(chunk_data.hashes), min(subsample_chunk, len(chunk_data.hashes)), replace=False
        )
        # pick indices from hashes, embeds, task_ids and flags
        data.hashes.extend(np.asarray(chunk_data.hashes)[indices].tolist())
        data.encodings.extend(np.asarray(chunk_data.encodings)[indices].tolist())
        data.task_ids.extend(np.asarray(chunk_data.task_ids)[indices].tolist())
        data.is_test.extend(np.asarray(chunk_data.is_test)[indices].tolist())

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

        if algo == "faiss":
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

            if use_constraints:
                D, I = index.search(training_embeds, int(n_clusters))
                centroids = balance_clusters(I, D, kmeans.centroids, training_embeds)
                # recompute centroids
                index.reset()
                index.add(centroids.astype("float32"))

            cluster_infos = ClusterInfos()

            # assignment
            for chunk in chunks:
                data = Encodings.load(chunk)

                embeds = np.asarray(data.encodings).astype("float32")

                if use_normalization:
                    embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

                if use_centering:
                    embeds = torch.from_numpy(embeds).T
                    embeds = embeds - centering.mm(embeds)
                    embeds = embeds.T.numpy().astype("float32")

                if use_pca > 0:
                    embeds = mat.apply(embeds)

                D, I = index.search(embeds, int(n_clusters))

                cluster_infos.hashes.extend(data.hashes)
                cluster_infos.task_names.extend(data.task_names)
                cluster_infos.cluster_ids.extend(torch.from_numpy(I[:, 0]).flatten().tolist())
                cluster_infos.is_test.extend(data.is_test)

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
