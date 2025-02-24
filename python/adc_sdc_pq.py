import numpy as np
import faiss

class PQIndex:
    def __init__(self, d=256, m=8, nbits=8):
        self.d = d
        self.m = m
        self.nbits = nbits
        self.k = 2 ** nbits  # 修正 `k` 计算
        self.pq = faiss.ProductQuantizer(d, m, nbits)
        self.codes = None

    def train(self, data):
        """训练 PQ 量化器"""
        assert data.shape[1] == self.d, f"输入数据维度 {data.shape[1]} 与设定维度 {self.d} 不匹配"
        self.pq.train(data)

    def encode(self, data):
        """编码数据"""
        return self.pq.compute_codes(data).astype(np.uint8)  # 确保返回 uint8

    def add(self, data):
        """添加数据到索引"""
        self.codes = self.encode(data).reshape(-1, self.m)  # 直接使用 `.astype(np.uint8)`
        print(f"codes shape: {self.codes.shape}")

    def search_adc(self, query, topk=10):
        """ADC 搜索"""
        query = query.reshape(-1)  # 确保 query 是 (d,)
        assert query.shape[0] == self.d, "查询向量维度错误"
        # subqueries：m * dsub
        subqueries = query.reshape(self.m, self.pq.dsub).astype(np.float32)

        # 获取 PQ 码本数据
        # centroids：m * k * dsub
        centroids = faiss.vector_to_array(self.pq.centroids).reshape(self.m, self.k, self.pq.dsub)

        # 计算距离表
        # D_table：m * k
        D_table = np.zeros((self.m, self.k), dtype=np.float32)
        for i in range(self.m):
            D_table[i] = np.linalg.norm(centroids[i] - subqueries[i], axis=1) ** 2

        sub_distance = D_table[np.arange(self.m), self.codes]
        distances = np.sum(sub_distance, axis=1)

        # 取前 topk 最近的索引
        idx = np.argsort(distances)[:topk]
        return idx, distances[idx]

    def search_sdc(self, query, topk=10):
        """SDC 搜索"""
        query = query.reshape(1, -1)  # 确保 query 是 (1, d)
        q_code = self.pq.compute_codes(query).astype(np.uint8).reshape(self.m)  # 转换成 (m,)

        # 计算码本距离表
        D_table = np.zeros((self.m, self.k, self.k), dtype=np.float32)
        centroids = faiss.vector_to_array(self.pq.centroids).reshape(self.m, self.k, self.pq.dsub)
        for i in range(self.m):
            D_table[i] = ((centroids[i, None] - centroids[i][:, None]) ** 2).sum(axis=2)

        # 计算数据库向量的总距离
        distances = D_table[np.arange(self.m), q_code, self.codes].sum(axis=1)  # 累加所有子空间
        # print(f"distance shape: {distances.shape}")
        idx = np.argsort(distances)[:topk]
        return idx, distances[idx]

# 使用示例
if __name__ == "__main__":
    topK = 10
    d = 256
    data = np.random.randn(10000, d).astype(np.float32)
    query = np.random.randn(1, d).astype(np.float32)

    pq_index = PQIndex(d=d, m=8, nbits=8)
    pq_index.train(data)
    pq_index.add(data)

    # 真实最近邻搜索
    distances = np.linalg.norm(data - query, axis=1)
    true_top = np.argsort(distances)[:topK]
    print(f"groud truth: {true_top.shape}")

    # 测试 ADC
    adc_result, _ = pq_index.search_adc(query, topK)
    print(f"adc: {adc_result.shape}")
    print(f"ADC Recall@{topK}: {len(np.intersect1d(true_top, adc_result)) / topK:.3f}")

    # 测试 SDC
    sdc_result, _ = pq_index.search_sdc(query, topK)
    print(f"sdc: {sdc_result.shape}")
    print(f"SDC Recall@{topK}: {len(np.intersect1d(true_top, sdc_result)) / topK:.3f}")
