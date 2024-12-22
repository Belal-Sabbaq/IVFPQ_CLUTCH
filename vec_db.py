from typing import Annotated
import numpy as np
import os
from IVFPQ import IVF_PQ

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self,  sub_clusters: int,d=70, m=10, k=50, probes=10000,
                 database_file_path="./assets/databases/saved_db_20m.dat",
                 index_file_path = "./assets/indexes/ivf_pq/index_20m_k50/",
                 new_db=True, db_size=None) -> None:
        self.D = d  # Vector dimension
        self.database_size = db_size
        self.db_path = database_file_path
        self.index_file_path = index_file_path

        # Default values for dynamic parameter adjustment
        self.k = k
        self.M = m
        self.Probes = probes
        self.SubClusters = sub_clusters

        # Dynamically adjust parameters based on database size
        if db_size is not None:
            if db_size <= 1_000_000:  # 1M
                self.k = 50
                self.M = 10
                self.Probes = 1000
                self.SubClusters = 256
                train_limit=500_000
                batchSize = 1000
            elif db_size <= 10_000_000:  # 10M
                self.k = 50
                self.M = 10
                self.Probes = 10000
                self.SubClusters = 4096
                train_limit=4_000_000
                batchSize = 5000
            elif db_size <= 15_000_000:  # 15M
                self.K = 70
                self.M = 10
                batchSize = 7000
                self.Probes = 10000
                self.SubClusters = 4096
                train_limit=2_000_000
            elif db_size <= 20_000_000:  # 20M
                self.K = 80
                self.M = 10
                self.Probes = 100000
                batchSize = 10000
                self.SubClusters = 4096
                train_limit=4_000_000
            else:  # Larger than 20M
                print("Warning: Dataset size exceeds predefined cases. Using default parameters.")
    
            # Logging for debugging
            print(f"Database size: {db_size}, K: {self.k}, M: {self.M}, Probes: {self.Probes}, SubClusters: {self.SubClusters}")
        if new_db:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.ivf_pq = IVF_PQ(self.D, self.k, self.Probes, self.M, self.SubClusters,self.index_file_path ,batch_size=batchSize)
            self.generate_database()
        else:
            
            self.ivf_pq = IVF_PQ(self.D, self.k, self.Probes, self.M, self.SubClusters, self.index_file_path,new_index=False)

    def generate_database(self) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((self.database_size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()
        print("Db Generated")

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()

        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function only loads one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this loads all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        self.ivf_pq.load_ivf_centroids()
        return self.ivf_pq.search(query, top_k)

    def _build_index(self):
        database = self.get_all_rows()
        self.ivf_pq.generate_ivf_pq_index(database)
