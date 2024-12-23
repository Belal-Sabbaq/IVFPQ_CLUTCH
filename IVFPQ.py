import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
from typing import Annotated

DIMENSION = 70

class IVF_PQ:
    def __init__(self, d: int, k: int, probes: int, m: int, sub_clusters: int,
                 index_file_path,
                 batch_size = 1000, 
                 train_limit=10**6, iterations=128, new_index=True) -> None:
        self.D = d                                              # Initial vector dimensions
        self.K = k                                              # Number of clusters for IVF
        self.K_ = probes                                        # Number of clusters retrieved
        self.M = m                                              # PQ subvector count
        self.D_ = d // m                                        # Dimensions per subvector
        self.sub_clusters = sub_clusters                       # Number of sub-clusters for PQ
        self.batch_size = batch_size 
        self.train_limit = train_limit
        self.iterations = iterations

        self.clusters_path = index_file_path + "/clusters"
        self.centroids_path =  index_file_path + "/centroids"
        self.pq_centroids_path = index_file_path + "/pq_centroids"
        
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

        # Using a small batch of data for k-means if dataset is large
        training_data, predicting_data = database[:self.train_limit], database[self.train_limit:]

        # Apply MiniBatchKMeans instead of kmeans2 for better memory management
        kmeans = MiniBatchKMeans(n_clusters=self.K, max_iter=self.iterations, init='k-means++', batch_size=self.batch_size)
        kmeans.fit(training_data)

        self.ivf_centroids = kmeans.cluster_centers_
        vectors = kmeans.predict(training_data)

        np.savetxt(self.centroids_path, self.ivf_centroids)

        # Parallelize cluster processing
        def process_cluster(i):
            ids = np.where(vectors == i)[0]  # Retrieve vector IDs
            cluster = database[ids]

            # Dynamically adjust sub-clusters based on cluster size
            sub_clusters = min(self.sub_clusters, cluster.shape[0])
            if sub_clusters != self.sub_clusters:
                print(f"Warning: Cluster {i} has fewer than {self.sub_clusters} points. Adjusting sub_clusters to {sub_clusters}")

            # PQ: Refining each cluster using MiniBatchKMeans for each sub-cluster
            pq_centroids = np.zeros((self.M, sub_clusters, self.D_))
            pq_codebook = np.zeros((self.M, cluster.shape[0]), dtype=np.uint32)

            for j in range(self.M):
                # Mini-batch k-means for PQ refinement
                pq_kmeans = MiniBatchKMeans(n_clusters=sub_clusters, max_iter=self.iterations, init='k-means++', batch_size=100)
                pq_kmeans.fit(cluster[:, j * self.D_:(j + 1) * self.D_])
                pq_centroids[j, :, :] = pq_kmeans.cluster_centers_
                pq_codebook[j, :] = pq_kmeans.predict(cluster[:, j * self.D_:(j + 1) * self.D_])

            cluster_index_path = os.path.join(self.clusters_path, f"cluster{i}")
            # Save IDs instead of vectors
            np.savez(cluster_index_path, pq_centroids=pq_centroids, pq_codebook=pq_codebook, ids=ids)

        # Use parallel processing to handle the clusters concurrently
        Parallel(n_jobs=-1)(delayed(process_cluster)(i) for i in range(self.K))

        print("All clusters processed.")

    def load_ivf_centroids(self):
        self.ivf_centroids = np.loadtxt(self.centroids_path)

    def search(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k: int):
        query = query.flatten()

        # Step 1: Find the nearest clusters using IVF centroids
        distances = self._compute_cosine_similarity(self.ivf_centroids, query)
        nearest_clusters = np.argsort(distances)[-self.K_:]

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
