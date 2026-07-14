import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # mac 常用中文字体
plt.rcParams['axes.unicode_minus'] = False 

# 加载数据
wine = load_wine()
X = wine.data
# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def find_elbow_k(k_values, inertias):
	"""使用“端点连线最大距离法”近似寻找肘部点。"""
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
		distance = abs(line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]) / line_norm
		distances.append(distance)

	best_index = int(np.argmax(distances))
	return int(k_values[best_index])

# 1) 肘部法寻找最优 K
k_values = np.arange(1, 11)
inertias = []

for k in k_values:
	model = KMeans(n_clusters=k, random_state=42, n_init=10)
	model.fit(X_scaled)
	inertias.append(model.inertia_)

best_k = find_elbow_k(k_values, inertias)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.axvline(best_k, color='red', linestyle='--', label=f'推荐 K = {best_k}')
plt.title('K-Means 肘部法：不同 K 对应的 SSE(Inertia)')
plt.xlabel('K 值')
plt.ylabel('SSE / Inertia')
plt.xticks(k_values)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2024150040邓瑞霖kmeans.png'), dpi=200)


# 2) 用最优 K 训练最终模型
final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = final_model.fit_predict(X_scaled)

wine_result = pd.DataFrame(X, columns=wine.feature_names)
wine_result['cluster'] = labels

cluster_counts = wine_result['cluster'].value_counts().sort_index()
print('肘部法推荐的最优 K 值：', best_k)
print('\n各簇样本数：')
print(cluster_counts)
print('\n最终模型 Inertia：', final_model.inertia_)

# 3) 使用 PCA 做二维可视化，方便观察聚类效果
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=35, alpha=0.85)
plt.title(f'K-Means 聚类结果可视化（K = {best_k}）')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(*scatter.legend_elements(), title='Cluster', loc='best')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2024150040邓瑞霖kmeans_clusters.png'), dpi=200)

wine_result.to_csv(os.path.join(OUTPUT_DIR, '2024150040邓瑞霖kmeans_result.csv'), index=False, encoding='utf-8-sig')

# 4) DBSCAN 实验：观察 eps 与 min_samples 对聚类和噪声点的影响
# 经过预实验，Wine 数据集在标准化后需要更大的 eps 才能形成有效簇
eps_list = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
min_samples_list = [2, 3, 4, 5, 8, 10]
dbscan_summary = []

for eps in eps_list:
	for ms in min_samples_list:
		db = DBSCAN(eps=eps, min_samples=ms)
		db_labels = db.fit_predict(X_scaled)
		# 计算簇数量（不计噪声标签 -1）
		n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
		n_noise = int((db_labels == -1).sum())

		# 计算 silhouette（仅当簇数 >=2 时）
		try:
			sil = float(silhouette_score(X_scaled, db_labels)) if n_clusters >= 2 else float('nan')
		except Exception:
			sil = float('nan')

		dbscan_summary.append({'eps': eps, 'min_samples': ms, 'n_clusters': n_clusters, 'n_noise': n_noise, 'silhouette': sil})

		# PCA 可视化并保存
		plt.figure(figsize=(6, 5))
		# 将噪声点标为灰色
		unique_labels = set(db_labels)
		colors = plt.cm.tab10.colors
		for lab in unique_labels:
			if lab == -1:
				col = 'lightgray'
				label_name = 'noise'
				marker = 'x'
			else:
				col = colors[int(lab) % len(colors)]
				label_name = f'cluster {lab}'
				marker = 'o'
			mask = (db_labels == lab)
			plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], label=label_name, s=30, alpha=0.8, marker=marker)

		plt.title(f'DBSCAN eps={eps}, min_samples={ms} (clusters={n_clusters}, noise={n_noise})')
		plt.xlabel('PCA 1')
		plt.ylabel('PCA 2')
		plt.legend(loc='best', fontsize='small')
		plt.tight_layout()
		fname = os.path.join(OUTPUT_DIR, f'2024150040邓瑞霖dbscan_eps{str(eps).replace(".","_")}_ms{ms}.png')
		plt.savefig(fname, dpi=200)
		plt.close()

# 保存 DBSCAN 结果汇总
pd.DataFrame(dbscan_summary).to_csv(os.path.join(OUTPUT_DIR, '2024150040邓瑞霖dbscan_summary.csv'), index=False, encoding='utf-8-sig')

# 5) KMeans 与 DBSCAN 的对比可视化（选取一个代表性参数集）
# 这里选择 eps=2.0, min_samples=3：可形成多个簇，同时保留一定数量噪声点，便于观察参数影响。
chosen_eps = 2.0
chosen_ms = 3
db_chosen = DBSCAN(eps=chosen_eps, min_samples=chosen_ms)
db_chosen_labels = db_chosen.fit_predict(X_scaled)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=35, alpha=0.85)
plt.title(f'KMeans (K={best_k})')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(*scatter.legend_elements(), title='Cluster', loc='best')

plt.subplot(1, 2, 2)
# DBSCAN 可视化：噪声为灰色
unique_labels = set(db_chosen_labels)
for lab in unique_labels:
	if lab == -1:
		col = 'lightgray'
		marker = 'x'
		label_name = 'noise'
	else:
		col = colors[int(lab) % len(colors)]
		marker = 'o'
		label_name = f'cluster {lab}'
	mask = (db_chosen_labels == lab)
	plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], label=label_name, s=35, alpha=0.85, marker=marker)

plt.title(f'DBSCAN (eps={chosen_eps}, min_samples={chosen_ms})')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2024150040邓瑞霖kmeans_dbscan_comparison.png'), dpi=200)
plt.close()
