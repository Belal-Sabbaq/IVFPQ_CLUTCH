from typing import Annotated
import numpy as np
import os
from scipy.cluster.vq import kmeans2, vq

DIMENSION = 70

class ProductQuantization:
    def __init__(self, d: int, m: int, k: int,
                 centroids_path="./assets/indexes/product_quantization/centroids",
                 codebook_path="./assets/indexes/product_quantization/codebook",
                 train_limit=10**6, iterations=128, new_index=True) -> None:
        self.D = d                                              # initial vector dimensions
        self.M = m                                              # new vector dimensions ( # of sub vectors)
        self.D_ = d // m                                        # sub vectors dimensions
        self.K = k                                              # number of clusters

        self.train_limit = train_limit
        self.iterations = iterations

        self.centroids_path = centroids_path
        self.codebook_path = codebook_path

        if new_index:
            if os.path.exists(self.centroids_path):
                os.remove(self.centroids_path)
            if os.path.exists(self.codebook_path):
                os.remove(self.codebook_path)

    def generate_product_quantization(self, database: np.ndarray) -> np.ndarray:
        training_data, predicting_data = database[:self.train_limit], database[self.train_limit:]

        centroids = np.zeros((self.M, self.K, self.D_))
        code_book = np.zeros((self.M, database.shape[0]), dtype=np.uint32)         # uint32 instead of float64

        for i in range(self.M):
            centroids[i, :, :], code_book[i, :self.train_limit] = kmeans2(training_data[:, i * self.D_: (i + 1) * self.D_], self.K,
                                                         minit='points', iter=self.iterations)

        if predicting_data.shape[0] > 0:
            for i in range(self.M):
                code_book[i, self.train_limit:], _ = vq(predicting_data[:, i * self.D_: (i + 1) * self.D_], centroids[i])

        self.centroids = centroids
        self.code_book = code_book.T

        np.savetxt(self.centroids_path, self.centroids.reshape(self.M * self.K, self.D_))
        np.savetxt(self.codebook_path, self.code_book, fmt="%d")

        return self.code_book

    def load_centroids(self):
        self.centroids = np.loadtxt(self.centroids_path).reshape((self.M, self.K, self.D_))

    def load_codebook(self):
        self.code_book = np.loadtxt(self.codebook_path)

    def search(self, query: Annotated[np.ndarray, (1, DIMENSION)], candidates: np.ndarray, top_k: int):
        ids = candidates[:, -1]
        vectors = candidates[:, :-1]

        distances = np.zeros((self.M, self.K))
        query = query.reshape(self.M, self.D_)

        for i in range(self.M):
            distances[i] = np.linalg.norm(self.centroids[i] - query[i], axis=1) ** 2

        n = vectors.shape[0]
        scores = np.zeros(n)
        for i in range(self.M):
            scores += distances[i, vectors[:, i]]

        top_vecs = np.argpartition(scores, top_k)[:top_k]

        return ids[top_vecs]






