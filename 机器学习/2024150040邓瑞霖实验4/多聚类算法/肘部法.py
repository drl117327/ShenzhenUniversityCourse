import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # mac 常用中文字体
plt.rcParams['axes.unicode_minus'] = False


def find_elbow_k(k_values, inertias):
	"""使用端点连线最大距离法近似寻找肘部点。"""
	points = np.column_stack([k_values, inertias])
	start = points[0]
	end = points[-1]
	line_vec = end - start
	line_norm = np.linalg.norm(line_vec)

	if line_norm == 0:
		return int(k_values[0])

	distances = []
	for point in points:
		point_vec = point - start
		# 2D 点到直线距离：|x1*y2 - y1*x2| / ||line||
		distance = abs(line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]) / line_norm
		distances.append(distance)

	return int(k_values[int(np.argmax(distances))])


def main():
	# 1. 读取 Wine 数据集并标准化
	wine = load_wine()
	X = wine.data
	X_scaled = StandardScaler().fit_transform(X)

	# 2. 计算 K=1~10 时的 SSE(Inertia)
	k_values = np.arange(1, 11)
	inertias = []

	for k in k_values:
		model = KMeans(n_clusters=k, random_state=42, n_init=10)
		model.fit(X_scaled)
		inertias.append(model.inertia_)

	# 3. 用肘部法估计最佳 K
	best_k = find_elbow_k(k_values, inertias)

	# 4. 绘图
	plt.figure(figsize=(8, 5))
	plt.plot(k_values, inertias, marker='o', linewidth=2)
	plt.axvline(best_k, color='red', linestyle='--', label=f'推荐 K = {best_k}')
	plt.title('Wine 数据集的肘部法曲线')
	plt.xlabel('K 值')
	plt.ylabel('簇内误差平方和 (SSE / Inertia)')
	plt.xticks(k_values)
	plt.grid(alpha=0.3)
	plt.legend()
	plt.tight_layout()

	out_path = os.path.join(OUTPUT_DIR, 'elbow_wine.png')
	plt.savefig(out_path, dpi=200)
	plt.show()

	# 5. 输出结果
	print('K 值范围:', list(k_values))
	print('对应 SSE(Inertia):', [round(v, 2) for v in inertias])
	print('根据肘部法推荐的最合适 K 值:', best_k)
	print(f'图像已保存到: {out_path}')


if __name__ == '__main__':
	main()
