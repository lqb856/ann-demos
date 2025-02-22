import faiss
from dataloader import *

from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Optional
import psutil

import numpy

class BaseANN(object):
    """Base class/interface for Approximate Nearest Neighbors (ANN) algorithms used in benchmarking."""

    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        pass

    def get_memory_usage(self) -> Optional[float]:
        """Returns the current memory usage of this ANN algorithm instance in kilobytes.

        Returns:
            float: The current memory usage in kilobytes (for backwards compatibility), or None if
                this information is not available.
        """

        return psutil.Process().memory_info().rss / 1024

    def fit(self, X: numpy.array) -> None:
        """Fits the ANN algorithm to the provided data. 

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            X (numpy.array): The data to fit the algorithm to.
        """
        pass

    def query(self, q: numpy.array, n: int) -> numpy.array:
        """Performs a query on the algorithm to find the nearest neighbors. 

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            q (numpy.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.

        Returns:
            numpy.array: An array of indices representing the nearest neighbors.
        """
        return []  # array of candidate indices

    def batch_query(self, X: numpy.array, n: int) -> None:
        """Performs multiple queries at once and lets the algorithm figure out how to handle it.

        The default implementation uses a ThreadPool to parallelize query processing.

        Args:
            X (numpy.array): An array of vectors to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return for each query.
        Returns: 
            None: self.get_batch_results() is responsible for retrieving batch result
        """
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self) -> numpy.array:
        """Retrieves the results of a batch query (from .batch_query()).

        Returns:
            numpy.array: An array of nearest neighbor results for each query in the batch.
        """
        return self.res

    def get_additional(self) -> Dict[str, Any]:
        """Returns additional attributes to be stored with the result.

        Returns:
            dict: A dictionary of additional attributes.
        """
        return {}

    def __str__(self) -> str:
        return self.name

class Faiss(BaseANN):
    def query(self, v, n):
        D, I = self.index.search(numpy.expand_dims(v, axis=0).astype(numpy.float32), n)
        return I[0]

    def batch_query(self, X, n):
        self.res = self.index.search(X.astype(numpy.float32), n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res

class FaissIVFPQfs(Faiss):
    def __init__(self, n_list):
        self._n_list = n_list

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        d = X.shape[1]
        faiss_metric = faiss.METRIC_L2
        factory_string = f"IVF{self._n_list},PQ{d//2}x4fs"
        index = faiss.index_factory(d, factory_string, faiss_metric)
        index.train(X)
        index.add(X)
        index_refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(X))
        self.base_index = index
        self.refine_index = index_refine

    def set_query_arguments(self, n_probe, k_reorder):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self._k_reorder = k_reorder
        self.base_index.nprobe = self._n_probe
        self.refine_index.k_factor = self._k_reorder
        if self._k_reorder == 0:
            self.index = self.base_index
        else:
            self.index = self.refine_index

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * self._n_list}  # noqa

    def __str__(self):
        return "FaissIVFPQfs(n_list=%d, n_probe=%d, k_reorder=%d)" % (self._n_list, self._n_probe, self._k_reorder)

if __name__ == "__main__":
    
    # 加载数据集
    base_vec = read_fvecs("/home/lqb/ann-dataset/SIFT1M/sift_base.fvecs")
    base_vec = np.ascontiguousarray(base_vec)
    print(base_vec.shape)
    query_vec = read_fvecs("/home/lqb/ann-dataset/SIFT1M/sift_query.fvecs")
    query_vec = np.ascontiguousarray(query_vec)
    print(query_vec.shape)
    ground_truth = read_ivecs_variable_k("/home/lqb/ann-dataset/SIFT1M/sift_groundtruth.ivecs")
    ground_truth = np.ascontiguousarray(ground_truth)
    print(ground_truth.shape)
    
    # 参数设定
    d = base_vec.shape[1]
    m = 32
    nbits = 8
    nlist = 16
    nprobe = 4
    topK = ground_truth.shape[1]
    iter = 10
    
    # 创建IVFPQ索引
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    
    # index = FaissIVFPQfs(nlist)
    
    # 训练并添加数据
    index.train(base_vec)
    index.add(base_vec)
    # index.fit(base_vec)
    # index.set_query_arguments(nprobe, 0)
    
    avg_recall = 0.0
    for id, query in enumerate(query_vec[:iter]):
        true_top_id = ground_truth[id]
        query = query.reshape(1, -1)
        distances, indices = index.search(query, k=topK)
        # indices = index.query(query, topK)
        recall = len(np.intersect1d(true_top_id, indices)) / topK
        avg_recall += recall
        print(f"Faiss ADC Recall@{topK}: {recall:.3f}")
    
    print(f"avg recall: {avg_recall / iter:.3f}")
    