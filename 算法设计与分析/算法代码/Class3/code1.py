from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Iterable
import argparse
import csv
import math
import random
import statistics

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TrialResult:
	floors: int
	eggs: int
	bruteforce_trials: int | None
	naive_dp_trials: int | None
	optimized_dp_trials: int


def brute_force_min_trials(floors: int, eggs: int) -> int:
	"""蛮力递归 + 记忆化：直接枚举首层投放楼层。"""
	if floors < 0 or eggs < 1:
		raise ValueError("floors must be >= 0 and eggs must be >= 1")

	@lru_cache(maxsize=None)
	def solve(f: int, e: int) -> int:
		if f == 0:
			return 0
		if f == 1:
			return 1
		if e == 1:
			return f

		best = math.inf
		for x in range(1, f + 1):
			broken = solve(x - 1, e - 1)
			not_broken = solve(f - x, e)
			worst = 1 + max(broken, not_broken)
			if worst < best:
				best = worst
		return int(best)

	return solve(floors, eggs)


def naive_dp_min_trials(floors: int, eggs: int) -> int:
	"""朴素动态规划 O(eggs * floors^2)。"""
	if floors < 0 or eggs < 1:
		raise ValueError("floors must be >= 0 and eggs must be >= 1")
	if floors == 0:
		return 0

	dp = [[0] * (floors + 1) for _ in range(eggs + 1)]
	for f in range(1, floors + 1):
		dp[1][f] = f
	for e in range(1, eggs + 1):
		dp[e][0] = 0
		dp[e][1] = 1 if floors >= 1 else 0

	for e in range(2, eggs + 1):
		for f in range(2, floors + 1):
			best = math.inf
			for x in range(1, f + 1):
				worst = 1 + max(dp[e - 1][x - 1], dp[e][f - x])
				if worst < best:
					best = worst
			dp[e][f] = int(best)
	return dp[eggs][floors]


def optimized_dp_min_trials(floors: int, eggs: int) -> int:
	"""优化动态规划：以“可覆盖楼层数”为状态。"""
	if floors < 0 or eggs < 1:
		raise ValueError("floors must be >= 0 and eggs must be >= 1")
	if floors == 0:
		return 0

	covered = [0] * (eggs + 1)
	moves = 0
	while covered[eggs] < floors:
		moves += 1
		for e in range(eggs, 0, -1):
			covered[e] = covered[e] + covered[e - 1] + 1
	return moves


def verify_small_cases(sample_size: int = 100, seed: int = 20260529) -> list[TrialResult]:
	"""随机生成小规模样例，验证蛮力法与 DP 结果一致。"""
	rng = random.Random(seed)
	results: list[TrialResult] = []
	for _ in range(sample_size):
		floors = rng.randint(0, 12)
		eggs = rng.randint(1, 5)
		bf = brute_force_min_trials(floors, eggs)
		naive = naive_dp_min_trials(floors, eggs)
		opt = optimized_dp_min_trials(floors, eggs)
		if not (bf == naive == opt):
			raise AssertionError(
				f"Mismatch for floors={floors}, eggs={eggs}: brute={bf}, naive={naive}, opt={opt}"
			)
		results.append(TrialResult(floors, eggs, bf, naive, opt))
	return results


def benchmark(
	cases: Iterable[tuple[int, int]],
	repeat: int = 5,
	seed: int = 20260529,
	include_naive: bool = True,
) -> list[dict[str, str]]:
	"""对不同规模进行计时测试。"""
	rng = random.Random(seed)
	rows: list[dict[str, str]] = []

	for floors, eggs in cases:
		# 固定同一组样例，减少随机波动。
		floors = int(floors)
		eggs = int(eggs)
		if eggs < 1:
			raise ValueError("eggs must be >= 1")

		naive_times: list[float] = []
		opt_times: list[float] = []
		bf_times: list[float] = []

		for _ in range(repeat):
			# 引入一点随机性，但保证每次可复现。
			_ = rng.random()

			if include_naive and floors <= 400:
				start = perf_counter()
				naive = naive_dp_min_trials(floors, eggs)
				naive_times.append(perf_counter() - start)
			else:
				naive = None

			start = perf_counter()
			opt = optimized_dp_min_trials(floors, eggs)
			opt_times.append(perf_counter() - start)

			if floors <= 12 and eggs <= 5:
				start = perf_counter()
				bf = brute_force_min_trials(floors, eggs)
				bf_times.append(perf_counter() - start)
				if bf != opt:
					raise AssertionError(
						f"Benchmark validation failed for floors={floors}, eggs={eggs}: bf={bf}, opt={opt}"
					)
			else:
				bf = None

		rows.append(
			{
				"floors": str(floors),
				"eggs": str(eggs),
				"bruteforce_avg_ms": f"{statistics.mean(bf_times) * 1000:.6f}" if bf_times else "",
				"naive_dp_avg_ms": f"{statistics.mean(naive_times) * 1000:.6f}" if naive_times else "",
				"optimized_dp_avg_ms": f"{statistics.mean(opt_times) * 1000:.6f}",
				"result": str(opt if opt is not None else ""),
			}
		)

	return rows


def benchmark_three_methods(
	eggs: int,
	max_floors: int,
	step: int,
	repeat: int = 5,
) -> list[dict[str, str]]:
	"""在同一规模下对比蛮力法、朴素DP、优化DP的运行时间。"""
	if eggs < 1:
		raise ValueError("eggs must be >= 1")
	if max_floors < 1:
		raise ValueError("max_floors must be >= 1")
	if step < 1:
		raise ValueError("step must be >= 1")

	rows: list[dict[str, str]] = []
	for floors in range(step, max_floors + 1, step):
		bf_times: list[float] = []
		naive_times: list[float] = []
		opt_times: list[float] = []

		for _ in range(repeat):
			start = perf_counter()
			bf = brute_force_min_trials(floors, eggs)
			bf_times.append(perf_counter() - start)

			start = perf_counter()
			naive = naive_dp_min_trials(floors, eggs)
			naive_times.append(perf_counter() - start)

			start = perf_counter()
			opt = optimized_dp_min_trials(floors, eggs)
			opt_times.append(perf_counter() - start)

			if not (bf == naive == opt):
				raise AssertionError(
					f"Mismatch for floors={floors}, eggs={eggs}: brute={bf}, naive={naive}, opt={opt}"
				)

		rows.append(
			{
				"floors": str(floors),
				"eggs": str(eggs),
				"bruteforce_avg_ms": f"{statistics.mean(bf_times) * 1000:.6f}",
				"naive_dp_avg_ms": f"{statistics.mean(naive_times) * 1000:.6f}",
				"optimized_dp_avg_ms": f"{statistics.mean(opt_times) * 1000:.6f}",
			}
		)

	return rows


def plot_three_methods(rows: list[dict[str, str]], output_png: Path) -> None:
	"""将三种方法的运行时间绘制为同一张折线图。"""
	if not rows:
		raise ValueError("rows is empty")

	floors = [int(r["floors"]) for r in rows]
	bruteforce_ms = [float(r["bruteforce_avg_ms"]) for r in rows]
	naive_ms = [float(r["naive_dp_avg_ms"]) for r in rows]
	optimized_ms = [float(r["optimized_dp_avg_ms"]) for r in rows]
	eggs = rows[0]["eggs"]

	output_png.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(9, 5.5))
	plt.plot(floors, bruteforce_ms, marker="o", linewidth=1.8, label="Brute Force + Memo")
	plt.plot(floors, naive_ms, marker="s", linewidth=1.8, label="Naive DP")
	plt.plot(floors, optimized_ms, marker="^", linewidth=2.2, label="Optimized DP")
	plt.title(f"Egg Dropping Runtime Comparison (eggs={eggs})")
	plt.xlabel("Floors")
	plt.ylabel("Average Runtime (ms)")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_png, dpi=180)
	plt.close()


def default_correctness_cases() -> list[tuple[int, int]]:
	"""用于正确性验证展示的固定样例。"""
	return [
		(5, 1),
		(6, 2),
		(10, 2),
		(12, 2),
		(14, 3),
		(20, 3),
		(25, 3),
		(30, 4),
	]


def build_correctness_rows(cases: Iterable[tuple[int, int]]) -> list[dict[str, str]]:
	"""生成正确性验证表：蛮力法、朴素DP、优化DP最终结果。"""
	rows: list[dict[str, str]] = []
	for floors, eggs in cases:
		bf = brute_force_min_trials(floors, eggs)
		naive = naive_dp_min_trials(floors, eggs)
		opt = optimized_dp_min_trials(floors, eggs)
		consistent = bf == naive == opt
		rows.append(
			{
				"floors": str(floors),
				"eggs": str(eggs),
				"bruteforce_result": str(bf),
				"naive_dp_result": str(naive),
				"optimized_dp_result": str(opt),
				"is_consistent": "True" if consistent else "False",
			}
		)
		if not consistent:
			raise AssertionError(
				f"Correctness mismatch: floors={floors}, eggs={eggs}, brute={bf}, naive={naive}, opt={opt}"
			)
	return rows


def plot_correctness_results(rows: list[dict[str, str]], output_png: Path) -> None:
	"""绘制三种方法最终结果对比图（柱状图）。"""
	if not rows:
		raise ValueError("rows is empty")

	labels = [f"f={r['floors']},e={r['eggs']}" for r in rows]
	bf_values = [int(r["bruteforce_result"]) for r in rows]
	naive_values = [int(r["naive_dp_result"]) for r in rows]
	opt_values = [int(r["optimized_dp_result"]) for r in rows]

	indices = list(range(len(rows)))
	width = 0.24

	output_png.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(11, 5.8))
	plt.bar([i - width for i in indices], bf_values, width=width, label="Brute Force + Memo")
	plt.bar(indices, naive_values, width=width, label="Naive DP")
	plt.bar([i + width for i in indices], opt_values, width=width, label="Optimized DP")
	plt.xticks(indices, labels, rotation=25, ha="right")
	plt.ylabel("Minimum Trials")
	plt.title("Correctness Validation: Final Results Comparison")
	plt.grid(True, axis="y", linestyle="--", alpha=0.35)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_png, dpi=180)
	plt.close()


def efficiency_small_cases() -> list[tuple[int, int]]:
	"""用于效率测试的小规模样例，保证蛮力法可运行。"""
	return [(floors, 2) for floors in (5, 6, 7, 8, 9, 10, 11, 12)]


def large_scale_cases() -> list[tuple[int, int]]:
	"""用于效率测试的大规模样例，按鸡蛋数分组绘图。"""
	return [
		(500, 2),
		(1000, 2),
		(5000, 2),
		(10000, 2),
		(100000, 2),
		(1000000, 2),
		(1000, 3),
		(10000, 3),
		(100000, 3),
		(1000000, 3),
	]


def build_efficiency_small_rows(repeat: int = 5) -> list[dict[str, str]]:
	"""生成小规模效率测试表：三种方法都记录。"""
	rows: list[dict[str, str]] = []
	for floors, eggs in efficiency_small_cases():
		bf_times: list[float] = []
		naive_times: list[float] = []
		opt_times: list[float] = []
		result = None
		for _ in range(repeat):
			start = perf_counter()
			bf = brute_force_min_trials(floors, eggs)
			bf_times.append(perf_counter() - start)

			start = perf_counter()
			naive = naive_dp_min_trials(floors, eggs)
			naive_times.append(perf_counter() - start)

			start = perf_counter()
			opt = optimized_dp_min_trials(floors, eggs)
			opt_times.append(perf_counter() - start)

			if not (bf == naive == opt):
				raise AssertionError(
					f"Efficiency mismatch for floors={floors}, eggs={eggs}: brute={bf}, naive={naive}, opt={opt}"
				)
			result = opt

		rows.append(
			{
				"floors": str(floors),
				"eggs": str(eggs),
				"bruteforce_avg_ms": f"{statistics.mean(bf_times) * 1000:.6f}",
				"naive_dp_avg_ms": f"{statistics.mean(naive_times) * 1000:.6f}",
				"optimized_dp_avg_ms": f"{statistics.mean(opt_times) * 1000:.6f}",
				"result": str(result),
			}
		)

	return rows


def plot_efficiency_small(rows: list[dict[str, str]], output_png: Path) -> None:
	"""绘制小规模效率测试折线图，比较三种方法耗时。"""
	if not rows:
		raise ValueError("rows is empty")

	floors = [int(r["floors"]) for r in rows]
	bruteforce_ms = [float(r["bruteforce_avg_ms"]) for r in rows]
	naive_ms = [float(r["naive_dp_avg_ms"]) for r in rows]
	optimized_ms = [float(r["optimized_dp_avg_ms"]) for r in rows]
	output_png.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(9, 5.5))
	plt.plot(floors, bruteforce_ms, marker="o", linewidth=1.8, label="Brute Force + Memo")
	plt.plot(floors, naive_ms, marker="s", linewidth=1.8, label="Naive DP")
	plt.plot(floors, optimized_ms, marker="^", linewidth=2.2, label="Optimized DP")
	plt.title("Efficiency Test on Small Inputs (eggs=2)")
	plt.xlabel("Floors")
	plt.ylabel("Average Runtime (ms)")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_png, dpi=180)
	plt.close()


def plot_large_scale_same_eggs(rows: list[dict[str, str]], eggs: int, output_png: Path) -> None:
	"""绘制大规模数据在相同鸡蛋数下的优化DP折线图。"""
	filtered = [row for row in rows if int(row["eggs"]) == eggs]
	if not filtered:
		raise ValueError(f"No rows for eggs={eggs}")

	floors = [int(r["floors"]) for r in filtered]
	optimized_ms = [float(r["optimized_dp_avg_ms"]) for r in filtered]
	output_png.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(9, 5.5))
	plt.plot(floors, optimized_ms, marker="o", linewidth=2.2, label=f"Optimized DP (eggs={eggs})")
	plt.title(f"Large-Scale Efficiency Test (eggs={eggs})")
	plt.xlabel("Floors")
	plt.ylabel("Average Runtime (ms)")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_png, dpi=180)
	plt.close()


def plot_theoretical_efficiency(rows: list[dict[str, str]], output_png: Path) -> None:
	"""绘制理论复杂度对比折线图。"""
	if not rows:
		raise ValueError("rows is empty")

	floors = [int(r["floors"]) for r in rows]
	brute_theory = [float(f * f) for f in floors]
	naive_theory = [float(f * f) for f in floors]
	optimized_theory = [float(r["result"]) for r in rows]

	output_png.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(9, 5.5))
	plt.plot(floors, brute_theory, marker="o", linewidth=1.8, label="Brute Force Theory O(f^2)")
	plt.plot(floors, naive_theory, marker="s", linewidth=1.8, label="Naive DP Theory O(f^2)")
	plt.plot(floors, optimized_theory, marker="^", linewidth=2.2, label="Optimized DP Theory O(m)")
	plt.yscale("log")
	plt.title("Theoretical Efficiency Comparison")
	plt.xlabel("Floors")
	plt.ylabel("Theoretical Growth Index (log scale)")
	plt.grid(True, linestyle="--", alpha=0.4, which="both")
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_png, dpi=180)
	plt.close()


def plot_actual_vs_theoretical_efficiency(rows: list[dict[str, str]], output_png: Path) -> None:
	"""绘制实际耗时与归一化理论趋势的对比折线图。"""
	if not rows:
		raise ValueError("rows is empty")

	floors = [int(r["floors"]) for r in rows]
	bruteforce_actual = [float(r["bruteforce_avg_ms"]) for r in rows]
	naive_actual = [float(r["naive_dp_avg_ms"]) for r in rows]
	optimized_actual = [float(r["optimized_dp_avg_ms"]) for r in rows]

	brute_theory_raw = [float(f * f) for f in floors]
	naive_theory_raw = [float(f * f) for f in floors]
	optimized_theory_raw = [float(r["result"]) for r in rows]

	def normalize(theory: list[float], actual: list[float]) -> list[float]:
		base_theory = theory[0]
		base_actual = actual[0]
		if base_theory == 0:
			raise ValueError("theoretical base is zero")
		scale = base_actual / base_theory
		return [value * scale for value in theory]

	brute_theory = normalize(brute_theory_raw, bruteforce_actual)
	naive_theory = normalize(naive_theory_raw, naive_actual)
	optimized_theory = normalize(optimized_theory_raw, optimized_actual)

	output_png.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(10, 5.8))
	plt.plot(floors, bruteforce_actual, marker="o", linewidth=1.8, label="Brute Force Actual")
	plt.plot(floors, brute_theory, linestyle="--", linewidth=1.8, label="Brute Force Theory")
	plt.plot(floors, naive_actual, marker="s", linewidth=1.8, label="Naive DP Actual")
	plt.plot(floors, naive_theory, linestyle="--", linewidth=1.8, label="Naive DP Theory")
	plt.plot(floors, optimized_actual, marker="^", linewidth=2.2, label="Optimized DP Actual")
	plt.plot(floors, optimized_theory, linestyle="--", linewidth=2.2, label="Optimized DP Theory")
	plt.title("Actual Runtime vs Theoretical Trend (Normalized)")
	plt.xlabel("Floors")
	plt.ylabel("Runtime / Normalized Theory")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend(ncol=2)
	plt.tight_layout()
	plt.savefig(output_png, dpi=180)
	plt.close()


def write_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


def default_cases() -> list[tuple[int, int]]:
	return [
		(5, 1),
		(10, 2),
		(20, 2),
		(50, 2),
		(100, 2),
		(200, 2),
		(500, 2),
		(1000, 2),
		(5000, 2),
		(10000, 2),
		(100000, 2),
		(1000000, 2),
		(1000, 3),
		(10000, 3),
		(100000, 3),
		(1000000, 3),
	]


def summarize(rows: list[dict[str, str]]) -> str:
	lines = []
	lines.append("floors eggs result optimized_ms naive_ms brute_ms")
	for row in rows:
		lines.append(
			" ".join(
				[
					row["floors"],
					row["eggs"],
					row["result"],
					row["optimized_dp_avg_ms"],
					row["naive_dp_avg_ms"] or "-",
					row["bruteforce_avg_ms"] or "-",
				]
			)
		)
	return "\n".join(lines)


def main() -> None:
	parser = argparse.ArgumentParser(description="Egg dropping experiment")
	parser.add_argument("--csv", default="output/benchmark_results.csv", help="output CSV path")
	parser.add_argument("--repeat", type=int, default=5, help="repeat count per benchmark case")
	parser.add_argument("--no-naive", action="store_true", help="skip naive DP timing")
	parser.add_argument("--summary", action="store_true", help="print brief summary")
	parser.add_argument("--plot-compare", action="store_true", help="plot runtime comparison of 3 methods")
	parser.add_argument("--compare-eggs", type=int, default=2, help="eggs for 3-method comparison")
	parser.add_argument("--compare-max-floors", type=int, default=120, help="max floors for 3-method comparison")
	parser.add_argument("--compare-step", type=int, default=10, help="floor step for 3-method comparison")
	parser.add_argument(
		"--compare-csv",
		default="output/runtime_comparison.csv",
		help="output CSV for 3-method comparison",
	)
	parser.add_argument(
		"--compare-png",
		default="output/runtime_comparison.png",
		help="output PNG for 3-method comparison",
	)
	parser.add_argument("--export-correctness", action="store_true", help="export correctness table and plot")
	parser.add_argument(
		"--correctness-csv",
		default="output/correctness_validation.csv",
		help="output CSV for correctness validation",
	)
	parser.add_argument(
		"--correctness-png",
		default="output/correctness_validation.png",
		help="output PNG for correctness validation",
	)
	parser.add_argument("--export-efficiency", action="store_true", help="export efficiency tables and plots")
	parser.add_argument(
		"--eff-small-csv",
		default="output/efficiency_small.csv",
		help="output CSV for small-input efficiency test",
	)
	parser.add_argument(
		"--eff-small-png",
		default="output/efficiency_small.png",
		help="output PNG for small-input efficiency test",
	)
	parser.add_argument(
		"--eff-large-csv",
		default="output/efficiency_large.csv",
		help="output CSV for large-input efficiency test",
	)
	parser.add_argument(
		"--eff-large-png-prefix",
		default="output/efficiency_large_eggs",
		help="output PNG prefix for large-input efficiency plots",
	)
	parser.add_argument(
		"--theory-png",
		default="output/theoretical_efficiency.png",
		help="output PNG for theoretical efficiency comparison",
	)
	parser.add_argument(
		"--actual-theory-png",
		default="output/actual_vs_theory.png",
		help="output PNG for actual vs theoretical comparison",
	)
	args = parser.parse_args()

	verify_small_cases(sample_size=100)

	if args.export_correctness:
		correctness_rows = build_correctness_rows(default_correctness_cases())
		write_csv(correctness_rows, Path(args.correctness_csv))
		plot_correctness_results(correctness_rows, Path(args.correctness_png))
		print(f"Saved correctness validation CSV to {args.correctness_csv}")
		print(f"Saved correctness validation plot to {args.correctness_png}")
		return

	if args.export_efficiency:
		small_rows = build_efficiency_small_rows(repeat=args.repeat)
		write_csv(small_rows, Path(args.eff_small_csv))
		plot_efficiency_small(small_rows, Path(args.eff_small_png))
		plot_theoretical_efficiency(small_rows, Path(args.theory_png))
		plot_actual_vs_theoretical_efficiency(small_rows, Path(args.actual_theory_png))

		large_rows = benchmark(large_scale_cases(), repeat=args.repeat, include_naive=False)
		write_csv(large_rows, Path(args.eff_large_csv))
		plot_large_scale_same_eggs(large_rows, 2, Path(f"{args.eff_large_png_prefix}_eggs2.png"))
		plot_large_scale_same_eggs(large_rows, 3, Path(f"{args.eff_large_png_prefix}_eggs3.png"))
		print(f"Saved small efficiency CSV to {args.eff_small_csv}")
		print(f"Saved small efficiency plot to {args.eff_small_png}")
		print(f"Saved theoretical efficiency plot to {args.theory_png}")
		print(f"Saved actual-vs-theory plot to {args.actual_theory_png}")
		print(f"Saved large efficiency CSV to {args.eff_large_csv}")
		print(f"Saved large efficiency plots to {args.eff_large_png_prefix}_eggs2.png and {args.eff_large_png_prefix}_eggs3.png")
		return

	if args.plot_compare:
		compare_rows = benchmark_three_methods(
			eggs=args.compare_eggs,
			max_floors=args.compare_max_floors,
			step=args.compare_step,
			repeat=args.repeat,
		)
		write_csv(compare_rows, Path(args.compare_csv))
		plot_three_methods(compare_rows, Path(args.compare_png))
		print(f"Saved runtime comparison CSV to {args.compare_csv}")
		print(f"Saved runtime comparison plot to {args.compare_png}")
		return

	rows = benchmark(default_cases(), repeat=args.repeat, include_naive=not args.no_naive)
	write_csv(rows, Path(args.csv))

	if args.summary:
		print(summarize(rows))
	else:
		print(f"Saved benchmark results to {args.csv}")


if __name__ == "__main__":
	main()
