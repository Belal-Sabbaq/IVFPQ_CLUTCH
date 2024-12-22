import faiss
from typing import Annotated
import numpy as np
from scipy.cluster.vq import kmeans2, vq
import os

DIMENSION = 70

class IVF_PQ:
    def __init__(self, d: int, k: int, probes: int, m: int, sub_clusters: int,
                 clusters_path="./assets/indexes/ivf_pq/clusters",
                 centroids_path="./assets/indexes/ivf_pq/centroids",
                 pq_centroids_path="./assets/indexes/ivf_pq/pq_centroids",
                 train_limit=10**6, iterations=128, new_index=True) -> None:
        self.D = d                                              # initial vector dimensions
        self.K = k                                              # number of clusters for IVF
        self.K_ = probes                                        # number of clusters retrieved
        self.M = m                                              # PQ subvector count
        self.D_ = d // m                                        # dimensions per subvector
        self.sub_clusters = sub_clusters                       # number of sub-clusters for PQ

        self.train_limit = train_limit
        self.iterations = iterations

        self.clusters_path = clusters_path
        self.centroids_path = centroids_path
        self.pq_centroids_path = pq_centroids_path
        
        if new_index:
            if os.path.exists(self.clusters_path):
                for filename in os.listdir(self.clusters_path):
                    file_path = os.path.join(self.clusters_path, filename)
                    os.remove(file_path)
            if os.path.exists(self.centroids_path):
                os.remove(self.centroids_path)
            if os.path.exists(self.pq_centroids_path):
                os.remove(self.pq_centroids_path)

    def generate_ivf_pq_index(self, database: np.ndarray):
        # IVF: Clustering the database into K clusters
        print("Kmeans Starting...")
        training_data, predicting_data = database[:self.train_limit], database[self.train_limit:]
        
        # Using FAISS Kmeans for IVF clustering
        kmeans = faiss.Kmeans(d=self.D, k=self.K, niter=self.iterations, gpu=True)
        kmeans.train(training_data)
        self.ivf_centroids = kmeans.centroids  # Get the centroids
        np.savetxt(self.centroids_path, self.ivf_centroids)
        print("Kmeans IVF Finished")
        
        # FAISS: Convert kmeans centroids into a FAISS index
        kmeans_index = faiss.IndexFlatL2(self.D)  # L2 distance index
        kmeans_index.add(self.ivf_centroids)

        # Move to GPU
        gpu_kmeans = faiss.index_cpu_to_gpu(self.res, 0, kmeans_index)

        # Continue with PQ refinement for each cluster...
        for i in range(self.K):
            # Identify points belonging to cluster `i`
            _, assignments = gpu_kmeans.search(database, 1)  # Assign all data points to nearest cluster
            ids = np.where(assignments[:, 0] == i)[0]  # Get indices for cluster `i`
            cluster = database[ids]
            sub_clusters = min(self.sub_clusters, cluster.shape[0])

            print(f"Cluster {i}: {cluster.shape[0]} points.")
            
            # PQ clustering
            pq_centroids = np.zeros((self.M, sub_clusters, self.D_))
            pq_codebook = np.zeros((self.M, cluster.shape[0]), dtype=np.uint32)

            for j in range(self.M):
                pq_kmeans = faiss.Kmeans(d=self.D_, k=sub_clusters, niter=self.iterations, gpu=True)
                pq_kmeans.train(cluster[:, j * self.D_:(j + 1) * self.D_])
                pq_centroids[j, :, :] = pq_kmeans.centroids
                _, pq_codebook[j, :] = pq_kmeans.index.search(cluster[:, j * self.D_:(j + 1) * self.D_], 1)

            # Save PQ centroids and codebook
            cluster_index_path = os.path.join(self.clusters_path, f"cluster{i}")
            np.savez(cluster_index_path, pq_centroids=pq_centroids, pq_codebook=pq_codebook, ids=ids)
        print("IVF-PQ Index Generated")

    def load_ivf_centroids(self):
        self.ivf_centroids = np.loadtxt(self.centroids_path)

    def search(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k: int):
        query = query.flatten()

        # Step 1: Find the nearest clusters using IVF centroids
        distances = self._compute_cosine_similarity(self.ivf_centroids, query)
        nearest_clusters = np.argsort(distances)[-self.K_:]
        print("IVF FINITO Searching")

        # Step 2: Retrieve candidates from nearest clusters
        candidates = []
        for cluster_id in nearest_clusters:
            cluster_index_path = os.path.join(self.clusters_path, f"cluster{cluster_id}.npz")
            cluster_data = np.load(cluster_index_path)

            pq_centroids = cluster_data['pq_centroids']
            pq_codebook = cluster_data['pq_codebook']
            ids = cluster_data['ids']

            # Step 3: Use PQ centroids for finer-grained search
            scores = np.zeros(len(ids))
            query_subvectors = query.reshape(self.M, self.D_)

            for m in range(self.M):
                distances = np.linalg.norm(pq_centroids[m] - query_subvectors[m], axis=1) ** 2
                scores += distances[pq_codebook[m]]

            candidates.append((ids, scores))

        # Aggregate candidates and sort by scores
        all_ids, all_scores = zip(*candidates)
        all_ids = np.concatenate(all_ids)
        all_scores = np.concatenate(all_scores)
        top_indices = np.argsort(all_scores)[:top_k]

        return all_ids[top_indices]

    def _compute_cosine_similarity(self, vec1, vec2):
        dot_product = vec1 @ vec2.T
        norm_vec1 = np.linalg.norm(vec1, axis=1)
        norm_vec2 = np.linalg.norm(vec2)
        norm = norm_vec1 * norm_vec2
        distances = dot_product / norm

        return distances