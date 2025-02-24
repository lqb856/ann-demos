import faiss
from python.dataloader import *

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


class FaissNSG(Faiss):  # 继承自已有的 Faiss 基类
    def __init__(self, R: int = 32, build_type: int = 0):
        """
        初始化 NSG 索引参数
        Args:
            R (int): 构建图时每个节点的邻居数（控制图的连通性）
            build_type (int): 图构建算法类型（0=FAISS默认，1=更精确但更慢）
        """
        self._R = R
        self._build_type = build_type
        self._search_L = 40  # 默认搜索参数
        self.index = None  # 实际索引对象

    def fit(self, X: numpy.array) -> None:
        """
        构建 NSG 索引并添加数据
        Args:
            X (numpy.array): 训练数据，形状为 [n_samples, n_features]
        """
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        d = X.shape[1]
        self.index = faiss.IndexNSGFlat(d, self._R)  # 初始化 NSG 索引

        # 配置图构建参数
        self.index.nsg.build_type = self._build_type  # 0=默认构建，1=更精确

        # 若需要数据归一化（例如使用余弦相似度时）
        # faiss.normalize_L2(X)

        # NSG 通常无需显式训练，直接添加数据
        self.index.add(X)

    def set_query_arguments(self, search_L: int, search_type: int = 0):
        """
        设置查询参数
        Args:
            search_L (int): 搜索时访问的节点数（越大召回率越高，但越慢）
            search_type (int): 搜索类型（0=双向搜索，1=单向搜索）
        """
        self._search_L = search_L
        self.index.nsg.search_L = self._search_L
        self.index.nsg.search_type = search_type

    def get_additional(self) -> Dict[str, Any]:
        """
        返回额外的统计信息（例如距离计算次数）
        """
        return {
            "search_L": self._search_L,
            "build_type": self._build_type,
            "R": self._R
        }

    def __str__(self) -> str:
        return f"FaissNSG(R={self._R}, build_type={self._build_type}, search_L={self._search_L})"

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
    reorder_factor = 0
    iter = 10
    
    # NSG
    neighbor_num = 32
    build_type = 0 # 0=默认构建（暴力搜索），1=NNDescent 算法构建
    search_L = 3 # 搜索时访问的节点数（越大召回率越高，但越慢）
    search_type = 0 # 0=双向搜索，1=单向搜索
    
    
    # 创建IVFPQ索引
    # 1. 使用原始的 IVFPQ
    # quantizer = faiss.IndexFlatL2(d)
    # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    # # 训练并添加数据
    # index.train(base_vec)
    # index.add(base_vec)
    
    # 2. 使用 Faiss 的工厂方法创建 IVFPQ
    # index = FaissIVFPQfs(nlist)
    # 训练索引
    # index.fit(base_vec)
    # index.set_query_arguments(nprobe, reorder_factor)
    
    # 3. 使用 NSG
    index = FaissNSG(neighbor_num, build_type)
    index.fit(base_vec)
    index.set_query_arguments(search_L=search_L, search_type=search_type)
    
    avg_recall = 0.0
    for id, query in enumerate(query_vec[:iter]):
        true_top_id = ground_truth[id]
        # query = query.reshape(1, -1)
        # distances, indices = index.search(query, k=topK)
        indices = index.query(query, topK)
        recall = len(np.intersect1d(true_top_id, indices)) / topK
        avg_recall += recall
        print(f"{index} Recall@{topK}: {recall:.3f}")
    
    print(f"avg recall: {avg_recall / iter:.3f}")
    