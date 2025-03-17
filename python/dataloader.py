import numpy as np

def read_fvecs(file_path):
    """高效读取 .fvecs 文件"""
    # 使用 numpy 从文件中读取数据
    data = np.fromfile(file_path, dtype=np.float32)
    
    if data.size == 0:
        return np.array([])  # 文件为空，返回空数组

    # 第一个整数表示向量的维度
    dim = int(data.view(np.int32)[0])
    
    # 每个向量包含一个维度信息（整数）和实际数据（dim 个 float）
    vector_size = 1 + dim
    
    # 计算向量的数量
    num_vectors = data.size // vector_size
    
    # 重塑数据为 (num_vectors, vector_size)
    data = data.reshape(num_vectors, vector_size)
    
    # 去除每个向量的第一个元素（维度信息），仅保留实际数据
    return data[:, 1:]

def read_fbin(file_path):
    """
    读取 .fbin 格式的向量文件
    
    参数：
        file_path: 文件路径（如 "base.1B.fbin"）
    
    返回：
        data: (N, dim) 的 float32 numpy 数组
    """
    with open(file_path, "rb") as f:
        # 读取维度（头部的 int32）
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        print(f"dim={dim}")
        
        # 读取剩余数据（全部为 float32）
        data = np.fromfile(f, dtype=np.float32)[1:]
        
        # 转换为 (N, dim) 矩阵
        data = data.reshape(dim, -1)
        return data
    
def read_ivecs(filename):
    """读取 .ivecs 文件并返回 (N, k) 的 int32 矩阵"""
    data = np.fromfile(filename, dtype=np.int32)
    k = data[0]  # 第一条记录的k值（假设所有记录的k相同）
    return data.reshape(-1, k + 1)[:, 1:]  # 跳过首位的k值

def read_ivecs_variable_k(filename):
    """读取允许k值变化的.ivecs文件"""
    data = np.fromfile(filename, dtype=np.int32)
    results = []
    ptr = 0  # 数据指针
    
    while ptr < len(data):
        k = data[ptr]  # 当前记录的k值
        ptr += 1
        ids = data[ptr:ptr + k].copy()  # 提取k个ID
        results.append(ids)
        ptr += k
    return np.vstack(results)  # 返回列表，每个元素是不同k值的数组