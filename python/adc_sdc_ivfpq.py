import numpy as np
import faiss

from python.dataloader import *

def unpack_pq_codes(packed_codes, nbits, m):
    """
    解包 FAISS bit-packed PQ 代码，恢复成 uint16 或 uint8 矩阵（支持 nbits > 8）。
    
    参数：
        packed_codes: 形状 (N, byte_per_code) 的 numpy uint8 数组（bit-packed 数据）
        nbits: 每个 PQ 子空间的编码 bit 数，如 4, 6, 8, 10, 12, 16
        m: 子空间个数（解包后形状应该是 (N, m)）
    
    返回：
        unpacked_codes: 形状 (N, m) 的 uint16 或 uint8 矩阵，每个值在 0 ~ (2^nbits - 1)
    """
    if nbits == 8:
        return packed_codes
    
    N, byte_per_code = packed_codes.shape  # bit-packed 矩阵形状
    total_bits = byte_per_code * 8  # 计算总 bit 数

    # 确保解包后的大小正确
    assert (total_bits // nbits) == m, f"bit 总数 ({total_bits}) 与 m ({m}) 不匹配"
    
    # 选择数据类型（避免 uint8 溢出）
    dtype = np.uint16 if nbits > 8 else np.uint8
    unpacked_codes = np.zeros((N, m), dtype=dtype)

    # 计算 bit-mask
    bit_mask = (1 << nbits) - 1  # 例如 nbits=10，bit_mask=0b1111111111（1023）

    for i in range(m):
        bit_offset = i * nbits  # 计算当前值的 bit 偏移
        byte_offset = bit_offset // 8  # 所属字节
        bit_shift = bit_offset % 8  # 需要右移的 bit 数
        
        # 取出 byte_offset 及后续字节（保证跨字节能完整取到）
        values = packed_codes[:, byte_offset].astype(np.uint16)  # 先转换成 uint16，避免溢出
        if bit_shift + nbits > 8:
            values |= packed_codes[:, byte_offset + 1].astype(np.uint16) << 8  # 取下一个字节
        if bit_shift + nbits > 16:
            values |= packed_codes[:, byte_offset + 2].astype(np.uint16) << 16  # 取第三个字节（用于 nbits > 16）

        # 右移 & 取 mask 得到 PQ 编码
        unpacked_codes[:, i] = (values >> bit_shift) & bit_mask

    return unpacked_codes


class IVFPQIndex:
    def __init__(self, d=256, m=8, nbits=8, nlist=100, nprobe=10, reorder_factor = 1):
        """
        d: 原始向量维度
        m: 子空间数量
        nbits: 每个子空间使用的比特数
        nlist: IVF 倒排列表的簇数
        nprobe: 搜索时探查的簇数
        """
        self.d = d
        self.m = m
        self.nbits = nbits
        self.k = 2 ** nbits  # 每个子空间的聚类中心数
        self.nlist = nlist
        self.nprobe = nprobe
        self.data_store = None
        self.reorder = False
        self.reorder_factor = int(reorder_factor)
        if self.reorder_factor > 1:
            self.reorder = True
        
        self.residual = False
        self.index_flat = None

        # PQ 量化器（对向量进行子量化）
        self.pq = faiss.ProductQuantizer(d, m, nbits)
        # IVF 聚类器的簇中心（后续用于分配数据）
        self.ivf_centroids = None

        # 倒排列表：字典 key 为簇 id，value 为该簇中所有数据点的 PQ 码（shape: (m,) per数据点）
        self.invlists = {i: [] for i in range(nlist)}
        # 对应数据点的原始编号
        self.invlists_ids = {i: [] for i in range(nlist)}

    def train(self, data):
        """
        训练阶段：
         - 使用全部数据训练 PQ 码表
         - 使用原始数据训练 IVF 聚类器（KMeans），得到 nlist 个簇中心
        """
        assert data.shape[1] == self.d, f"输入数据维度 {data.shape[1]} 与设定维度 {self.d} 不匹配"

        # 训练 PQ（码表）
        self.pq.train(data)

        # 训练 KMeans 得到 IVF 聚类中心
        kmeans = faiss.Kmeans(self.d, self.nlist, niter=20, verbose=False)
        kmeans.train(data)
        self.ivf_centroids = kmeans.centroids  # shape: (nlist, d)
        assert self.ivf_centroids.shape == (self.nlist, self.d)
        self.residual = False
        self.index_flat = None
        
    def train_residual(self, data):
        """
        训练阶段（使用残差进行量化）：
        1. 训练 IVF（KMeans），得到 nlist 个簇中心
        2. 计算残差（原始数据 - 最近的簇中心）
        3. 用残差数据训练 PQ（码表）
        """
        assert data.shape[1] == self.d, f"输入数据维度 {data.shape[1]} 与设定维度 {self.d} 不匹配"

        # 训练 KMeans 得到 IVF 聚类中心
        kmeans = faiss.Kmeans(self.d, self.nlist, niter=20, verbose=False)
        kmeans.train(data)
        self.ivf_centroids = kmeans.centroids  # shape: (nlist, d)
        assert self.ivf_centroids.shape == (self.nlist, self.d)

        # 计算残差
        index_flat = faiss.IndexFlatL2(self.d)
        index_flat.add(self.ivf_centroids)
        _, cluster_assignments = index_flat.search(data, 1)  # 选出最近的簇
        cluster_assignments = cluster_assignments.flatten()

        residuals = data - self.ivf_centroids[cluster_assignments]  # 计算残差
        assert residuals.shape == data.shape

        # 训练 PQ（码表）使用残差数据
        self.pq.train(residuals)
        self.residual = True
        self.index_flat = index_flat

    def add(self, data):
        """
        添加数据：
         - 使用原始数据对每个向量分配最近的 IVF 簇中心（用 IndexFlatL2 计算距离）
         - 对每个数据点进行 PQ 编码
         - 将编码及数据编号存入对应簇的倒排列表中
        """
        N = data.shape[0]
        # 使用 IndexFlatL2 计算每个数据点到簇中心的距离
        index_flat = None
        if self.index_flat == None:
            index_flat = faiss.IndexFlatL2(self.d)
            index_flat.add(self.ivf_centroids)  # IVF 簇中心作为“数据库”
            self.index_flat = index_flat
        else:
            index_flat = self.index_flat 
        distances, assignments = index_flat.search(data, 1)  # assignments shape: (N, 1)

        # PQ 编码（返回 uint8 数组，形状 (N, m)）
        # 注意，编码时也要使用残差进行编码！！
        if self.residual == True:
            residuals = data - self.ivf_centroids[assignments.flatten()]  # 计算残差
            codes = self.pq.compute_codes(residuals)  # 对残差进行编码
        else:
            codes = self.pq.compute_codes(data)
        # codes = unpack_pq_codes(codes, self.nbits, self.m)
        codes = codes.reshape(-1, self.m)
        
        # 将每个数据点分配到对应簇的倒排列表中
        for i, cluster in enumerate(assignments[:, 0]):
            self.invlists[cluster].append(codes[i])
            self.invlists_ids[cluster].append(i)
        for i in range(self.nlist):
            print(f"cluster {i} has {len(self.invlists_ids[i])} vecs")
            
        if self.reorder == True:
            self.data_store = data

    def search_adc(self, query, topk=10):
        """
        ADC 搜索：
         - 先计算查询向量与 IVF 簇中心的距离，选择最近的 nprobe 个簇
         - 在这些簇的倒排列表中，利用 PQ 码本构造距离表，
           对候选数据计算 ADC 距离（即：对每个子空间，取查询与码本中心距离，然后根据候选 PQ 码累加）
         - 返回候选数据的原始编号及其距离
        """
        search_k = topk * self.reorder_factor
        
        # 确保 query 是二维 (1, d)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # 对查询使用 IVF 聚类器：计算查询与每个簇中心的距离，选出 nprobe 个簇
        _, cluster_assignments = self.index_flat.search(query, self.nprobe)  # shape: (1, nprobe)
        selected_clusters = cluster_assignments[0]
        assert len(selected_clusters) == self.nprobe

        # 从倒排列表中收集候选数据的 PQ 编码和对应原始编号
        candidate_codes_list = []
        candidate_ids_list = []
        for cluster in selected_clusters:
            if len(self.invlists[cluster]) > 0:
                codes_array = np.array(self.invlists[cluster], dtype=np.uint8)  # shape: (n_cluster, m)
                candidate_codes_list.append(codes_array)
                candidate_ids_list.append(np.array(self.invlists_ids[cluster]))
        if len(candidate_codes_list) == 0:
            return np.array([]), np.array([])

        candidate_codes = np.concatenate(candidate_codes_list, axis=0)  # (N_candidates, m)
        candidate_ids = np.concatenate(candidate_ids_list, axis=0)      # (N_candidates,)
        assert candidate_codes.shape[0] == candidate_ids.shape[0]
        # print(f"ADC Num cantidate to search: {candidate_codes.shape[0]}")

        # 构造 ADC 距离表：
        # 将查询向量转为 (m, dsub)
        q = query.flatten()  # (d,)
        subqueries = q.reshape(self.m, self.pq.dsub).astype(np.float32)
        # 获取 PQ 码表 centroids，形状转换为 (m, k, dsub)
        centroids = faiss.vector_to_array(self.pq.centroids).reshape(self.m, self.k, self.pq.dsub)
        D_table = np.zeros((self.m, self.k), dtype=np.float32)
        for i in range(self.m):
            D_table[i] = np.linalg.norm(centroids[i] - subqueries[i], axis=1) ** 2

        # 对每个候选数据点，其 ADC 距离为：
        # distance = sum_{i=0}^{m-1} D_table[i, candidate_codes[c, i]]
        adc_distances = np.sum(D_table[np.arange(self.m)[:, None], candidate_codes.T], axis=0)  # (N_candidates,)

        # 选择 topk 最近的候选
        sorted_idx = np.argsort(adc_distances)
        top_idx = sorted_idx[:search_k]
        if self.reorder:
            return self.rerank(query, candidate_ids[top_idx], topk)
        return candidate_ids[top_idx], adc_distances[top_idx]
    
    def search_adc_residual(self, query, topk=10):
        """
        ADC 搜索：
         - 先计算查询向量与 IVF 簇中心的距离，选择最近的 nprobe 个簇
         - 再求出 query 在每个类里面的残差
         - 在这些簇的倒排列表中，利用 PQ 码本构造距离表，
           对候选数据计算 ADC 距离（即：对每个子空间，取查询与码本中心距离，然后根据候选 PQ 码累加）
         - 返回候选数据的原始编号及其距离
        """
        assert self.residual == True
        search_k = topk * self.reorder_factor
        # 确保 query 是二维 (1, d)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # 对查询使用 IVF 聚类器：计算查询与每个簇中心的距离，选出 nprobe 个簇
        _, cluster_assignments = self.index_flat.search(query, self.nprobe)  # shape: (1, nprobe)
        selected_clusters = cluster_assignments[0]
        assert len(selected_clusters) == self.nprobe

        # 从倒排列表中收集候选数据的 PQ 编码和对应原始编号
        candidate_codes_list = []
        candidate_ids_list = []
        for cluster in selected_clusters:
            if len(self.invlists[cluster]) > 0:
                codes_array = np.array(self.invlists[cluster], dtype=np.uint8)  # shape: (n_cluster, m)
                candidate_codes_list.append(codes_array)
                candidate_ids_list.append(np.array(self.invlists_ids[cluster]))
        if len(candidate_codes_list) == 0:
            return np.array([]), np.array([])

        # 构造 ADC 距离表：
        # 将查询向量转为 (m, dsub)
        q = query.flatten()  # (d,)
        distances = []
        # 共享码表
        centroids = faiss.vector_to_array(self.pq.centroids).reshape(self.m, self.k, self.pq.dsub)
        for idx, cluster in enumerate(selected_clusters):
            q_residual = q - self.ivf_centroids[cluster]
            subqueries = q_residual.reshape(self.m, self.pq.dsub).astype(np.float32)
            # 获取 PQ 码表 centroids，形状转换为 (m, k, dsub)
            D_table = np.zeros((self.m, self.k), dtype=np.float32)
            for i in range(self.m):
                D_table[i] = np.linalg.norm(centroids[i] - subqueries[i], axis=1) ** 2

            # 对每个候选数据点，其 ADC 距离为：
            # distance = sum_{i=0}^{m-1} D_table[i, candidate_codes[c, i]]
            # print(f"d_table: {D_table.shape}, candidate_codes: {candidate_codes_list[idx].shape}, arrange: {np.arange(self.m)[:, None].shape}")
            adc_distances = np.sum(D_table[np.arange(self.m)[:, None], candidate_codes_list[idx].T], axis=0)  # (N_candidates,)
            assert adc_distances.shape[0] == len(self.invlists_ids[cluster])
            distances.append(adc_distances)
            

        # 选择 topk 最近的候选
        candidate_ids = np.concatenate(candidate_ids_list, axis=0)      # (N_candidates,)
        distances = np.concatenate(distances, axis=0) # (N_candidates,)
        assert candidate_ids.shape[0] == distances.shape[0]
        sorted_idx = np.argsort(distances)
        top_idx = sorted_idx[:search_k]
        if self.reorder:
            return self.rerank(query, candidate_ids[top_idx], topk)
        return candidate_ids[top_idx], distances[top_idx]

    def search_sdc(self, query, topk=10):
        """
        SDC 搜索：
         - 与 ADC 搜索类似，但先对查询进行 PQ 编码（得到 q_code），
           再利用预计算的子量化器之间距离表计算距离。
         - 距离计算公式：distance = sum_{i=0}^{m-1} D_table[i, q_code[i], candidate_codes[c, i]]
        """
        search_k = topk * self.reorder_factor
        # 确保 query 是二维 (1, d)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # IVF 阶段：选出 nprobe 个最近簇
        _, cluster_assignments = self.index_flat.search(query, self.nprobe)
        selected_clusters = cluster_assignments[0]
        assert len(selected_clusters) == self.nprobe

        # 从倒排列表中收集候选数据的 PQ 编码和对应原始编号
        candidate_codes_list = []
        candidate_ids_list = []
        for cluster in selected_clusters:
            if len(self.invlists[cluster]) > 0:
                codes_array = np.array(self.invlists[cluster], dtype=np.uint8)
                candidate_codes_list.append(codes_array)
                candidate_ids_list.append(np.array(self.invlists_ids[cluster]))
        if len(candidate_codes_list) == 0:
            return np.array([]), np.array([])

        candidate_codes = np.concatenate(candidate_codes_list, axis=0)  # (N_candidates, m)
        candidate_ids = np.concatenate(candidate_ids_list, axis=0)      # (N_candidates,)
        assert candidate_codes.shape[0] == candidate_ids.shape[0]
        # print(f"SDC Num cantidate to search: {candidate_codes.shape[0]}")

        # 对查询向量进行 PQ 编码，得到 q_code (m,)
        q_code = self.pq.compute_codes(query).astype(np.uint8)
        # q_code = unpack_pq_codes(q_code, self.nbits, self.m)
        # print(f"Quantized Query Code Shape: {q_code.shape}")
        q_code = q_code.reshape(self.m)
        
        # 预计算子量化器内部的距离表：
        # 对每个子空间 i，计算其所有聚类中心之间的距离，
        # 得到 D_table[i] 形状 (k, k)
        D_table = np.zeros((self.m, self.k, self.k), dtype=np.float32)
        centroids = faiss.vector_to_array(self.pq.centroids).reshape(self.m, self.k, self.pq.dsub)
        for i in range(self.m):
            D_table[i] = ((centroids[i, None] - centroids[i][:, None]) ** 2).sum(axis=2)

        # 对每个候选数据，其 SDC 距离为：
        # distance = sum_{i=0}^{m-1} D_table[i, q_code[i], candidate_codes[c, i]]
        sdc_distances = np.sum(D_table[np.arange(self.m)[:, None], q_code[:, None], candidate_codes.T], axis=0)
        # sdc_distances1 = D_table[np.arange(self.m), q_code, candidate_codes].sum(axis=1)
        # assert np.array_equal(sdc_distances, sdc_distances1)

        sorted_idx = np.argsort(sdc_distances)
        top_idx = sorted_idx[:search_k]
        if self.reorder:
            return self.rerank(query, candidate_ids[top_idx], topk)
        return candidate_ids[top_idx], sdc_distances[top_idx]
    
    def search_sdc_residual(self, query, topk=10):
        """
        SDC 搜索：
         - 与 ADC 搜索类似，但先对查询进行 PQ 编码（得到 q_code），
           再利用预计算的子量化器之间距离表计算距离。
         - 距离计算公式：distance = sum_{i=0}^{m-1} D_table[i, q_code[i], candidate_codes[c, i]]
        """
        search_k = topk * self.reorder_factor
        assert self.residual == True
        # 确保 query 是二维 (1, d)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # IVF 阶段：选出 nprobe 个最近簇
        _, cluster_assignments = self.index_flat.search(query, self.nprobe)
        selected_clusters = cluster_assignments[0]
        assert len(selected_clusters) == self.nprobe

        # 从倒排列表中收集候选数据的 PQ 编码和对应原始编号
        candidate_codes_list = []
        candidate_ids_list = []
        for cluster in selected_clusters:
            if len(self.invlists[cluster]) > 0:
                codes_array = np.array(self.invlists[cluster], dtype=np.uint8)
                candidate_codes_list.append(codes_array)
                candidate_ids_list.append(np.array(self.invlists_ids[cluster]))
        if len(candidate_codes_list) == 0:
            return np.array([]), np.array([])

        distances = []
        centroids = faiss.vector_to_array(self.pq.centroids).reshape(self.m, self.k, self.pq.dsub)
        D_table = np.zeros((self.m, self.k, self.k), dtype=np.float32)
        for i in range(self.m):
            D_table[i] = ((centroids[i, None] - centroids[i][:, None]) ** 2).sum(axis=2)
        for idx, cluster in enumerate(selected_clusters):
            # 先求残差
            q_residual = query - self.ivf_centroids[cluster]
            # 对查询向量进行 PQ 编码，得到 q_code (m,)
            q_code = self.pq.compute_codes(q_residual).astype(np.uint8)
            # q_code = unpack_pq_codes(q_code, self.nbits, self.m)
            # print(f"Quantized Query Code Shape: {q_code.shape}")
            q_code = q_code.reshape(self.m)

            # 对每个候选数据，其 SDC 距离为：
            # distance = sum_{i=0}^{m-1} D_table[i, q_code[i], candidate_codes[c, i]]
            sdc_distances = np.sum(D_table[np.arange(self.m)[:, None], q_code[:, None], candidate_codes_list[idx].T], axis=0)
            # sdc_distances1 = D_table[np.arange(self.m), q_code, candidate_codes_list[i]].sum(axis=1)
            # assert np.array_equal(sdc_distances, sdc_distances1)
            distances.append(sdc_distances)

        candidate_ids = np.concatenate(candidate_ids_list, axis=0)      # (N_candidates,)
        distances = np.concatenate(distances, axis=0)      # (N_candidates,)
        assert candidate_ids.shape[0] == distances.shape[0]
        sorted_idx = np.argsort(distances)
        top_idx = sorted_idx[:search_k]
        if self.reorder:
            return self.rerank(query, candidate_ids[top_idx], topk)
        return candidate_ids[top_idx], distances[top_idx]
    
    def rerank(self, query, candidate_ids, topK):
        """
        对候选ID进行精确重排序：
        1. 根据候选ID获取原始向量
        2. 计算查询与每个候选向量的L2距离
        3. 按距离排序，返回排序后的ID和距离
        """
        # 确保query是二维数组
        if query.ndim == 1:
            query = query.reshape(1, -1)
        # 获取候选向量
        candidate_vectors = self.data_store[candidate_ids]
        # 计算精确距离
        distances = np.linalg.norm(candidate_vectors - query, axis=1) ** 2  # L2距离平方
        # 排序
        sorted_indices = np.argsort(distances)[:topK]
        sorted_ids = candidate_ids[sorted_indices]
        sorted_distances = distances[sorted_indices]
        return sorted_ids, sorted_distances
    
if __name__ == "__main__":
    
    # 加载数据集
    base_vec = read_fvecs("/home/lqb/ann-dataset/SIFT1M/sift/sift_base.fvecs")
    print(base_vec.shape)
    query_vec = read_fvecs("/home/lqb/ann-dataset/SIFT1M/sift/sift_query.fvecs")
    print(query_vec.shape)
    ground_truth = read_ivecs_variable_k("/home/lqb/ann-dataset/SIFT1M/sift/sift_groundtruth_100.ivecs")
    print(ground_truth.shape)
    
    # 参数设定
    d = base_vec.shape[1]
    m = 32
    nbits = 8
    nlist = 16
    nprobe = 4
    topK = ground_truth.shape[1]
    iter = 10
    residual = False
    reorder_factor = 1

    # 构建并训练 IVF-PQ 索引
    index = IVFPQIndex(d=d, m=m, nbits=nbits, nlist=nlist, nprobe=nprobe, reorder_factor=reorder_factor)
    if residual: 
        index.train_residual(base_vec)
    else:
        index.train(base_vec)
    # index.train_residual(base_vec)
    index.add(base_vec)

    avg_recall_adc = 0.0
    avg_recall_sdc = 0.0
    for id, query in enumerate(query_vec[:iter]):
        # 暴力计算真实最近邻（基于欧式距离）
        
        true_top_id = ground_truth[id]
        # print(f"Ground Truth Distance@{topK}: {true_distances[true_top_id]}")
        # print(f"Ground Truth Top@{topK}: {true_top_id}")

        # 使用 ADC 搜索
        if residual: 
            adc_ids, adc_dists = index.search_adc_residual(query, topk=topK)
        else:
            adc_ids, adc_dists = index.search_adc(query, topk=topK)
        # print(f"ADC Distance@{topK}: {adc_dists}")
        adc_recall = len(np.intersect1d(true_top_id, adc_ids)) / topK
        print(f"ADC Recall@{topK}: {adc_recall:.3f}")
        avg_recall_adc += adc_recall

        # 使用 SDC 搜索
        if residual: 
            sdc_ids, sdc_dists = index.search_sdc_residual(query, topk=topK)
        else:
            sdc_ids, sdc_dists = index.search_sdc(query, topk=topK)
        # print(f"SDC Distance@{topK}: {sdc_dists}")
        sdc_recall = len(np.intersect1d(true_top_id, sdc_ids)) / topK
        print(f"SDC Recall@{topK}: {sdc_recall:.3f}")
        avg_recall_sdc += sdc_recall
        
    print(f"avg adc recall: {avg_recall_adc / iter}")
    print(f"avg sdc recall: {avg_recall_sdc / iter}")