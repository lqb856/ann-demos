import heapq
import numpy as np
from collections import defaultdict

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

class NSGIndex:
    def __init__(self, data, knn_k=5, mst=True, max_neighbors=16):
        """
        完整NSG索引实现，包含建图过程
        
        参数:
            data: 数据集(numpy数组)
            knn_k: 初始KNN邻居数
            mst: 是否添加MST边保证连通性
            max_neighbors: 节点最大邻居数
        """
        self.data = data
        self.knn_k = knn_k
        self.max_neighbors = max_neighbors
        self.neighbors = self._build_graph(mst)
    
    def _build_graph(self, mst):
        """三层建图流程：KNN图 -> MST增强 -> 邻居优化"""
        # 第一阶段：构建初始KNN图
        knn_graph = self._build_knn_graph()
        
        # 第二阶段：构建MST并增强连通性
        if mst:
            mst_edges = self._build_mst()
            knn_graph = self._merge_graphs(knn_graph, mst_edges)
        
        # 第三阶段：邻居优化和剪枝
        return self._prune_neighbors(knn_graph)

    def _build_knn_graph(self):
        """暴力搜索构建KNN图"""
        n = len(self.data)
        graph = [[] for _ in range(n)]
        
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = euclidean_distance(self.data[i], self.data[j])
                    distances.append((dist, j))
            distances.sort()
            graph[i] = [j for _, j in distances[:self.knn_k]]
        return graph

    def _build_mst(self):
        """Prim算法构建最小生成树"""
        n = len(self.data)
        visited = [False]*n
        min_dist = [float('inf')]*n
        parent = [-1]*n
        
        # 从节点0开始构建
        min_dist[0] = 0
        heap = [(0, 0)]
        
        while heap:
            dist, u = heapq.heappop(heap)
            if visited[u]: continue
            visited[u] = True
            
            for v in range(n):
                if v == u: continue
                new_dist = euclidean_distance(self.data[u], self.data[v])
                if not visited[v] and new_dist < min_dist[v]:
                    min_dist[v] = new_dist
                    parent[v] = u
                    heapq.heappush(heap, (new_dist, v))
        
        # 构建双向MST边
        mst_edges = [[] for _ in range(n)]
        for v in range(1, n):
            u = parent[v]
            if u != -1:
                mst_edges[u].append(v)
                mst_edges[v].append(u)
        return mst_edges

    def _merge_graphs(self, knn_graph, mst_edges):
        """合并KNN图和MST边"""
        merged = [[] for _ in range(len(knn_graph))]
        for i in range(len(knn_graph)):
            # 合并时保持原始顺序：KNN在前，MST在后
            combined = knn_graph[i] + mst_edges[i]
            # 去重并保持插入顺序
            seen = set()
            merged[i] = []
            for j in combined:
                if j not in seen and j != i:
                    seen.add(j)
                    merged[i].append(j)
        return merged

    def _prune_neighbors(self, graph):
        """邻居优化：按距离排序并截断"""
        pruned = [[] for _ in range(len(graph))]
        for i in range(len(graph)):
            # 按距离排序
            neighbors = sorted(graph[i], 
                key=lambda x: euclidean_distance(self.data[i], self.data[x]))
            # 截断到最大邻居数
            pruned[i] = neighbors[:self.max_neighbors]
        return pruned

    def search(self, query_vec, enterpoints, max_hops, topK=3, beam_width=50):
        """
        支持多入口点的搜索。
        注意，最大跳数太少时可能导致搜索不到目标 topK 数量的向量。
        
        参数:
            query_vec: 查询向量
            enterpoints: 入口点列表（支持单个或多个）
            max_hops: 最大跳数限制
            topK: 返回点数
            beam_width: 候选集大小, 至少为topK大小
        """
        beam_width = max(beam_width, topK)
        beam = []
        visited = {}
        
        # 初始化入口点（支持单/多入口）
        enterpoints = [enterpoints] if isinstance(enterpoints, int) else enterpoints
        for ep in enterpoints:
            dist = euclidean_distance(query_vec, self.data[ep])
            self._add_to_beam(beam, (dist, ep, 0), beam_width)
            visited[ep] = (dist, 0)
        
        while beam:
            current_dist, current_node, hops = beam.pop(0)
            
            if hops >= max_hops:
                continue
                
            for neighbor in self.neighbors[current_node]:
                neighbor_dist = euclidean_distance(query_vec, self.data[neighbor])
                new_hops = hops + 1
                
                # 剪枝条件
                if neighbor in visited:
                    exist_dist, exist_hops = visited[neighbor]
                    if neighbor_dist >= exist_dist and new_hops >= exist_hops:
                        continue
                
                # 更新记录（允许更优路径覆盖）
                if neighbor not in visited or neighbor_dist < exist_dist:
                    visited[neighbor] = (neighbor_dist, new_hops)
                    self._add_to_beam(beam, (neighbor_dist, neighbor, new_hops), beam_width)
        
        # 直接返回前topK个（beam已有序）
        return [dis for dis, _, _ in beam[:topK]], [node for _, node, _ in beam[:topK]]

    def _add_to_beam(self, beam, element, max_size):
        """
        维护有序候选队列：
        1. 插入元素并保持按距离升序排列
        2. 控制队列长度不超过max_size
        """
        # 二分查找插入位置
        left, right = 0, len(beam)
        while left < right:
            mid = (left + right) // 2
            if beam[mid][0] < element[0]:
                left = mid + 1
            else:
                right = mid
        beam.insert(left, element)
        
        # 控制队列长度
        if len(beam) > max_size:
            beam.pop()  # 移除距离最大的元素