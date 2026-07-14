"""
排序推荐算法扩展实验 (第3、4、5题)
"""

import time
import math
import random
import heapq
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Item = tuple[int, float]  # (user_index, similarity)

# ==========================================
# 工具函数及旧有算法保留
# ==========================================
def cosine_similarity_matrix(users: np.ndarray, target: np.ndarray) -> np.ndarray:
    """计算 users 中每个用户向量与 target 的余弦相似度。"""
    target_norm = np.linalg.norm(target)
    user_norms = np.linalg.norm(users, axis=1)
    denom = np.clip(user_norms * target_norm, 1e-12, None)
    sims = users @ target / denom
    return sims

def heap_sort_desc(items: list[Item]) -> list[Item]:
    """基准：堆排序（全量排序后取截断）"""
    heap: list[tuple[float, int]] = [(-sim, idx) for idx, sim in items]
    heapq.heapify(heap)
    ordered: list[Item] = []
    while heap:
        neg_sim, idx = heapq.heappop(heap)
        ordered.append((idx, -neg_sim))
    return ordered

# ==========================================
# 优化算法：Top-k Min-Heap (第4题)
# ==========================================
def optimized_top_k_min_heap(items: list[Item], k: int) -> list[Item]:
    """
    优化算法：使用大小为 k 的最小堆维护当前找到的 k 个最大相似度。
    时间复杂度：O(n log k)
    """
    if k <= 0:
        return []
    
    # 堆中存储结构为 (sim, idx)，因 heapq 是最小堆，这样堆顶即为这 k 个之中最小的一个
    heap: list[tuple[float, int]] = []
    
    for idx, sim in items:
        if len(heap) < k:
            heapq.heappush(heap, (sim, idx))
        else:
            if sim > heap[0][0]:
                heapq.heapreplace(heap, (sim, idx))
                
    # 提出结果并按降序排序
    res: list[Item] = []
    while heap:
        s, i = heapq.heappop(heap)
        res.append((i, s))
    res.reverse()
    return res

# ==========================================
# 稀疏向量逻辑 (第5题)
# ==========================================
def generate_sparse_vectors(n: int, d: int, s: int) -> list[dict[int, float]]:
    """生成 n 个稀疏向量，总维度 d，非零元素个数 s。以字典存储 {index: value}"""
    docs = []
    for _ in range(n):
        indices = random.sample(range(d), s)
        vec = {idx: random.random() for idx in indices}
        docs.append(vec)
    return docs

def compute_sparse_sim(users_sparse: list[dict[int, float]], target_sparse: dict[int, float], d: int) -> list[float]:
    """使用稀疏计算法：将目标向量转为密集结构供 O(1) 查询，用户向量保持稀疏遍历"""
    target_dense = np.zeros(d, dtype=np.float64)
    target_norm_sq = 0.0
    for k, v in target_sparse.items():
        target_dense[k] = v
        target_norm_sq += v * v
    target_norm = math.sqrt(target_norm_sq)
    
    sims = []
    for user in users_sparse:
        dot = 0.0
        user_norm_sq = 0.0
        for k, v in user.items():
            dot += v * target_dense[k]
            user_norm_sq += v * v
        denom = max(math.sqrt(user_norm_sq) * target_norm, 1e-12)
        sims.append(dot / denom)
    return sims

# ==========================================
# 各部分实验执行
# ==========================================

def run_part3_impact_of_k(out_dir: Path):
    """3. 分析固定 n 下不同 k 值的效率影响"""
    print("\n--- 实验3：分析相同规模下不同的 k 值对基准算法效率的影响 ---")
    n = 10000
    dim = 50
    k_list = [10, 50, 100, 500, 1000, 5000, 10000]
    samples = 10
    rng = np.random.default_rng(2026)
    times = []
    
    users = rng.random((n, dim), dtype=np.float64)
    target = rng.random(dim, dtype=np.float64)
    sims = cosine_similarity_matrix(users, target)
    items = [(i, float(sims[i])) for i in range(len(sims))]
    
    for k in k_list:
        t_list = []
        for _ in range(samples):
            st = time.perf_counter()
            ranked = heap_sort_desc(items)
            res = ranked[:k]
            t_list.append((time.perf_counter() - st) * 1000)
        avg_t = np.mean(t_list)
        times.append(avg_t)
        print(f"n={n}, k={k:5d} -> 耗时 {avg_t:.4f} ms")
        
    plt.figure()
    plt.plot(k_list, times, marker='s', color='#d62728')
    plt.title("Impact of k on Full Sort Efficiency (n=10000)")
    plt.xlabel("k value")
    plt.ylabel("Time (ms)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(out_dir / "impact_of_k.png", dpi=180)
    plt.close()

def run_part4_optimized_topk(out_dir: Path):
    """4. 基准算法（全排序）与 优化算法（堆Top-k）对比"""
    print("\n--- 实验4：优化算法(Heap Top-k)与基准算法实测比较 ---")
    n_values = np.linspace(1000, 20000, 6).astype(int).tolist()
    k = 50
    dim = 50
    rng = np.random.default_rng(1234)
    samples = 15
    
    time_base = []
    time_opt = []
    
    for n in n_values:
        users = rng.random((n, dim), dtype=np.float64)
        target = rng.random(dim, dtype=np.float64)
        sims = cosine_similarity_matrix(users, target)
        items = [(i, float(sims[i])) for i in range(len(sims))]
        
        # 测基准
        tb = []
        for _ in range(samples):
            st = time.perf_counter()
            _ = heap_sort_desc(items)[:k]
            tb.append((time.perf_counter() - st) * 1000)
        time_base.append(np.mean(tb))
        
        # 测优化
        to = []
        for _ in range(samples):
            st = time.perf_counter()
            _ = optimized_top_k_min_heap(items, k)
            to.append((time.perf_counter() - st) * 1000)
        time_opt.append(np.mean(to))
        
        print(f"n={n}, 基准耗时: {time_base[-1]:.4f} ms | 优化耗时(Top-{k}): {time_opt[-1]:.4f} ms")
        
    plt.figure()
    plt.plot(n_values, time_base, marker='o', label="Baseline (Full Heap Sort)")
    plt.plot(n_values, time_opt, marker='^', label="Optimized (Min-Heap Top-k)")
    plt.title(f"Baseline vs Optimized Top-k (k={k})")
    plt.xlabel("Input Size n")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(out_dir / "optimized_vs_baseline.png", dpi=180)
    plt.close()

def run_part5_sparse_vectors(out_dir: Path):
    """5. 稀疏向量相似度计算效率随 s 的变化"""
    print("\n--- 实验5：稀疏向量不同 s 对计算相似度效率的影响 ---")
    n = 2000
    d = 10000  # 总维度假设很大
    s_values = [5, 10, 50, 100, 500, 1000]
    samples = 10
    
    times = []
    
    for s in s_values:
        t_list = []
        for _ in range(samples):
            # 生成测试数据
            users_sparse = generate_sparse_vectors(n, d, s)
            target_sparse = generate_sparse_vectors(1, d, s)[0]
            
            st = time.perf_counter()
            _ = compute_sparse_sim(users_sparse, target_sparse, d)
            t_list.append((time.perf_counter() - st) * 1000)
            
        avg_t = np.mean(t_list)
        times.append(avg_t)
        print(f"d={d}, n={n}, 非零元数 s={s:4d} -> 耗时 {avg_t:.4f} ms")
        
    plt.figure()
    plt.plot(s_values, times, marker='D', color="#2ca02c")
    plt.title("Efficiency of Sparse Vector Similarity (n=2000, d=10000)")
    plt.xlabel("Number of non-zeros (s)")
    plt.ylabel("Time (ms)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(out_dir / "sparse_efficiency_vs_s.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_part3_impact_of_k(out_dir)
    run_part4_optimized_topk(out_dir)
    run_part5_sparse_vectors(out_dir)
