from typing import Annotated
import numpy as np
from scipy.cluster.vq import kmeans2, vq
import os


DIMENSION = 70

class InvertedFlatIndex:

    def __init__(self, d: int, k: int, probes: int,
                 clusters_path="./assets/indexes/inverted_flat_index/clusters",
                 centroids_path="./assets/indexes/inverted_flat_index/centroids",
                 train_limit=10**6, iterations=128, new_index=True) -> None:
        self.D = d                                              # initial vector dimensions
        self.K = k                                              # number of clusters
        self.K_ = probes                                        # number of clusters retrieved

        self.train_limit = train_limit
        self.iterations = iterations

        self.clusters_path = clusters_path
        self.centroids_path = centroids_path

        if new_index:
            if os.path.exists(self.clusters_path):
                for filename in os.listdir(self.clusters_path):
                    file_path = os.path.join(self.clusters_path, filename)
                    os.remove(file_path)
            if os.path.exists(self.centroids_path):
                os.remove(self.centroids_path)

    def generate_inverted_flat_index(self, database: np.ndarray, codebook: np.ndarray):
        training_data, predicting_data = database[:self.train_limit], database[self.train_limit:]

        self.centroids, vectors = kmeans2(training_data, self.K, minit='points', iter=self.iterations)
        np.savetxt(self.centroids_path, self.centroids)

        for i in range(self.K):
            ids, = np.where(vectors == i)
            cluster = np.column_stack((codebook[ids], ids))
            np.savetxt(self.clusters_path + f"/cluster{i}", cluster, fmt="%d")

        if predicting_data.shape[0] > 0:
            vectors, _ = vq(predicting_data, self.centroids)

            for i in range(self.K):
                ids, = np.where(vectors == i)
                cluster = np.column_stack((codebook[ids], ids))

                pre_cluster = np.loadtxt(self.clusters_path + f"/cluster{i}", dtype=int)
                post_cluster = np.vstack((pre_cluster, cluster))
                np.savetxt(self.clusters_path + f"/cluster{i}", post_cluster, fmt="%d")

    def load_centroids(self):
        self.centroids = np.loadtxt(self.centroids_path)

    def search(self, query: Annotated[np.ndarray, (1, DIMENSION)], m: int):
        distances = self._compute_cosine_similarity(self.centroids, query[0])
        nearest_clusters = np.argsort(distances)[-self.K_:]

        candidates = np.empty((0, m + 1))
        for i in nearest_clusters:
            loaded_cluster = np.loadtxt(self.clusters_path + f"/cluster{int(i)}", dtype=int)
            candidates = np.append(candidates, loaded_cluster, axis=0)

        candidates = candidates.astype(int)

        return candidates

    def _compute_cosine_similarity(self, vec1, vec2):
        dot_product = vec1 @ vec2.T
        norm_vec1 = np.linalg.norm(vec1, axis=1)
        norm_vec2 = np.linalg.norm(vec2)
        norm = norm_vec1 * norm_vec2
        distances = dot_product / norm

        return distances