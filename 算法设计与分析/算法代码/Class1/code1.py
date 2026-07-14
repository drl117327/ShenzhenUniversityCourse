"""
基于排序的推荐算法实验

任务说明：
1. 计算目标用户与所有用户的相似度（余弦相似度）。
2. 使用不同排序算法对用户按相似度降序排序，取前 k 个用户。
3. 在不同输入规模 n 下进行 20 次随机实验，统计平均运行时间。
4. 固定 k，绘制“平均运行时间-输入规模 n”的关系曲线。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable
import csv
import math

import matplotlib.pyplot as plt
import numpy as np


# =========================
# 排序算法实现（按相似度降序）
# =========================
Item = tuple[int, float]  # (user_index, similarity)


def bubble_sort_desc(items: list[Item]) -> list[Item]:
	"""冒泡排序（降序）。"""
	arr = items[:]
	n = len(arr)
	for i in range(n):
		swapped = False
		for j in range(0, n - i - 1):
			if arr[j][1] < arr[j + 1][1]:
				arr[j], arr[j + 1] = arr[j + 1], arr[j]
				swapped = True
		if not swapped:
			break
	return arr


def merge_sort_desc(items: list[Item]) -> list[Item]:
	"""归并排序（降序）。"""

	def merge(left: list[Item], right: list[Item]) -> list[Item]:
		merged: list[Item] = []
		i = j = 0
		while i < len(left) and j < len(right):
			if left[i][1] >= right[j][1]:
				merged.append(left[i])
				i += 1
			else:
				merged.append(right[j])
				j += 1
		if i < len(left):
			merged.extend(left[i:])
		if j < len(right):
			merged.extend(right[j:])
		return merged

	if len(items) <= 1:
		return items[:]

	mid = len(items) // 2
	left = merge_sort_desc(items[:mid])
	right = merge_sort_desc(items[mid:])
	return merge(left, right)


def quick_sort_desc(items: list[Item]) -> list[Item]:
	"""快速排序（降序，三路划分，递归版）。"""
	if len(items) <= 1:
		return items[:]

	pivot = items[len(items) // 2][1]
	greater = [x for x in items if x[1] > pivot]
	equal = [x for x in items if x[1] == pivot]
	less = [x for x in items if x[1] < pivot]
	return quick_sort_desc(greater) + equal + quick_sort_desc(less)


def heap_sort_desc(items: list[Item]) -> list[Item]:
	"""堆排序（借助堆实现降序）。"""
	# 构建最大堆：通过存储 (-similarity, index) 到最小堆实现
	import heapq

	heap: list[tuple[float, int]] = [(-sim, idx) for idx, sim in items]
	heapq.heapify(heap)
	ordered: list[Item] = []
	while heap:
		neg_sim, idx = heapq.heappop(heap)
		ordered.append((idx, -neg_sim))
	return ordered


SORT_ALGOS: dict[str, Callable[[list[Item]], list[Item]]] = {
	"bubble_sort": bubble_sort_desc,
	"merge_sort": merge_sort_desc,
	"quick_sort": quick_sort_desc,
	"heap_sort": heap_sort_desc,
}


# =========================
# 推荐算法核心
# =========================
def cosine_similarity_matrix(users: np.ndarray, target: np.ndarray) -> np.ndarray:
	"""计算 users 中每个用户向量与 target 的余弦相似度。"""
	target_norm = np.linalg.norm(target)
	user_norms = np.linalg.norm(users, axis=1)
	denom = np.clip(user_norms * target_norm, 1e-12, None)
	sims = users @ target / denom
	return sims


def recommend_top_k(
	users: np.ndarray,
	target: np.ndarray,
	k: int,
	sort_fn: Callable[[list[Item]], list[Item]],
) -> list[Item]:
	"""计算相似度并排序，返回前 k 个最相似用户。"""
	sims = cosine_similarity_matrix(users, target)
	items: list[Item] = [(i, float(sims[i])) for i in range(len(sims))]
	ranked = sort_fn(items)
	return ranked[:k]


# =========================
# 实验与可视化
# =========================
@dataclass
class ExperimentConfig:
	dim: int = 50
	samples: int = 20
	n_min: int = 200
	n_max: int = 6000
	n_points: int = 12
	k_values: tuple[int, ...] = (1, 5, 10, 20)
	seed: int = 20260312


def uniform_n_values(n_min: int, n_max: int, n_points: int) -> list[int]:
	"""均匀取值的输入规模。"""
	values = np.linspace(n_min, n_max, n_points)
	ints = np.unique(np.rint(values).astype(int))
	return ints.tolist()


def run_benchmark(config: ExperimentConfig) -> dict[int, dict[str, dict[int, float]]]:
	"""执行基准测试，返回平均耗时（ms）。

	返回结构：
	{
	  k: {
		algo_name: {
		  n: avg_ms
		}
	  }
	}
	"""
	rng = np.random.default_rng(config.seed)
	n_values = uniform_n_values(config.n_min, config.n_max, config.n_points)
	results: dict[int, dict[str, dict[int, float]]] = {
		k: {algo: {} for algo in SORT_ALGOS} for k in config.k_values
	}

	for k in config.k_values:
		for n in n_values:
			# 对每个算法分别统计 20 次随机样本的耗时
			time_pool: dict[str, list[float]] = {algo: [] for algo in SORT_ALGOS}

			for _ in range(config.samples):
				users = rng.random((n, config.dim), dtype=np.float64)
				target = rng.random(config.dim, dtype=np.float64)

				for algo_name, algo_fn in SORT_ALGOS.items():
					start = perf_counter()
					topk = recommend_top_k(users, target, k, algo_fn)
					elapsed_ms = (perf_counter() - start) * 1000.0
					# 防止解释器优化（确保结果被“使用”）
					if len(topk) != min(k, n):
						raise RuntimeError("top-k 结果长度异常")
					time_pool[algo_name].append(elapsed_ms)

			for algo_name in SORT_ALGOS:
				results[k][algo_name][n] = float(np.mean(time_pool[algo_name]))

	return results


def save_csv(results: dict[int, dict[str, dict[int, float]]], output_file: Path) -> None:
	"""保存实验结果到 CSV。"""
	output_file.parent.mkdir(parents=True, exist_ok=True)
	with output_file.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["k", "algorithm", "n", "avg_time_ms"])
		for k, algo_map in results.items():
			for algo_name, n_map in algo_map.items():
				for n, avg_ms in sorted(n_map.items()):
					writer.writerow([k, algo_name, n, f"{avg_ms:.6f}"])


def plot_results(results: dict[int, dict[str, dict[int, float]]], out_dir: Path) -> None:
	"""对每个 k 绘制一张曲线图（包含实测曲线与冒泡排序理论曲线）。"""
	out_dir.mkdir(parents=True, exist_ok=True)

	for k, algo_map in results.items():
		plt.figure(figsize=(8.5, 5.2))
		for algo_name, n_map in algo_map.items():
			n_values = sorted(n_map.keys())
			times = [n_map[n] for n in n_values]
			plt.plot(n_values, times, marker="o", linewidth=1.8, label=f"Measured: {algo_name}")
			
			# 特别为冒泡排序计算并画出理论 O(n^2) 曲线
			if algo_name == "bubble_sort":
				# 以中间的一个 n 为基准点，映射理论运行时间
				base_idx = len(n_values) // 2
				base_n = n_values[base_idx]
				base_time = times[base_idx]
				
				# 理论上的时间应与 n^2 成正比，即 T(n) = C * n^2 -> C = base_time / (base_n**2)
				c = base_time / (base_n ** 2)
				theoretical_times = [c * (n ** 2) for n in n_values]
				
				plt.plot(n_values, theoretical_times, linestyle="--", color="black", 
						 linewidth=1.5, label="Theoretical: bubble_sort $O(n^2)$")

		plt.title(f"Runtime vs n (k={k}, 20 samples)")
		plt.xlabel("Input Size n")
		plt.ylabel("Average Runtime (ms)")
		plt.grid(True, linestyle="--", alpha=0.4)
		plt.legend()
		plt.tight_layout()
		plt.savefig(out_dir / f"runtime_k_{k}.png", dpi=180)
		plt.close()


def plot_nlogn_results(results: dict[int, dict[str, dict[int, float]]], out_dir: Path) -> None:
	"""单独绘制 O(n log n) 算法（归并、快排、堆排）的运行时间与理论时间对比图。"""
	out_dir.mkdir(parents=True, exist_ok=True)
	
	for k, algo_map in results.items():
		plt.figure(figsize=(9.5, 6))
		
		nlogn_algos = ["merge_sort", "quick_sort", "heap_sort"]
		colors = {"merge_sort": "blue", "quick_sort": "green", "heap_sort": "red"}
		
		for algo_name in nlogn_algos:
			if algo_name not in algo_map:
				continue
			n_map = algo_map[algo_name]
			n_values = sorted(n_map.keys())
			times = [n_map[n] for n in n_values]
			
			# 实测曲线
			plt.plot(n_values, times, marker="o", markersize=4, linewidth=1.5,
					 label=f"Measured: {algo_name}", color=colors[algo_name])
			
			# 理论 O(n log n) 拟合曲线
			# 以中间点的规模计算常数 C
			base_idx = len(n_values) // 2 + 2 
			base_n = n_values[base_idx]
			base_time = times[base_idx]
			
			c = base_time / (base_n * math.log2(base_n))
			theoretical_times = [c * (n * math.log2(n)) for n in n_values]
			
			# 理论曲线（虚线）
			plt.plot(n_values, theoretical_times, linestyle="--", linewidth=1.5,
					 color=colors[algo_name], alpha=0.5,
					 label=r"Theory $O(n \log n)$: " + algo_name)

		plt.title(f"O(n log n) Algorithms Runtime vs Theory (k={k}, 20 samples)")
		plt.xlabel("Input Size n")
		plt.ylabel("Average Runtime (ms)")
		plt.grid(True, linestyle="--", alpha=0.4)
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.tight_layout()
		plt.savefig(out_dir / f"nlogn_comparison_k_{k}.png", dpi=180)
		plt.close()


def print_summary(results: dict[int, dict[str, dict[int, float]]]) -> None:
	"""在终端输出简要结果。"""
	print("\n===== 实验汇总（单位：ms） =====")
	for k, algo_map in results.items():
		print(f"\n[k={k}]")
		for algo_name, n_map in algo_map.items():
			n_values = sorted(n_map)
			first_n = n_values[0]
			last_n = n_values[-1]
			print(
				f"  {algo_name:15s}: n={first_n} -> {n_map[first_n]:.4f}, "
				f"n={last_n} -> {n_map[last_n]:.4f}"
			)


def main() -> None:
	config = ExperimentConfig()
	base_dir = Path(__file__).resolve().parent
	out_dir = base_dir / "output"

	print("开始运行实验，请稍候...")
	results = run_benchmark(config)
	save_csv(results, out_dir / "benchmark_results.csv")
	plot_results(results, out_dir)
	plot_nlogn_results(results, out_dir)
	print_summary(results)

	print("\n实验完成。输出文件：")
	print(f"- {out_dir / 'benchmark_results.csv'}")
	for k in config.k_values:
		print(f"- {out_dir / f'runtime_k_{k}.png'}")
		print(f"- {out_dir / f'nlogn_comparison_k_{k}.png'}")


if __name__ == "__main__":
	main()
