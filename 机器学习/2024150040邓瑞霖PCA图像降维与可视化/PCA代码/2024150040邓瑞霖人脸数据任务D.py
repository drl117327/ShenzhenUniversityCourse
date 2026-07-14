import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

matplotlib.rcParams['font.sans-serif'] = ['Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False


# 加载数据
olivetti = fetch_olivetti_faces(shuffle=True, random_state=42, data_home='./数据集')
X = olivetti.data  # (400, 4096)
print('数据形状:', X.shape)

# 拟合全主成分（最多400个）
pca = PCA(n_components=400)
pca.fit(X)

explained_ratio = pca.explained_variance_ratio_
cumulative_ratio = np.cumsum(explained_ratio)
k_values = np.arange(1, len(cumulative_ratio) + 1)

# 计算达到90%累计解释率所需主成分数
threshold = 0.90
k_90 = int(np.argmax(cumulative_ratio >= threshold) + 1)
print(f'达到90%累计方差解释率所需主成分数: {k_90}')
print(f'对应累计解释率: {cumulative_ratio[k_90 - 1]:.6f}')

# 绘图：k 与累计方差解释率
plt.figure(figsize=(8.5, 5.2))
plt.plot(k_values, cumulative_ratio, color='royalblue', linewidth=1.8)
plt.axhline(y=threshold, color='red', linestyle='--', label='90%阈值')
plt.axvline(x=k_90, color='green', linestyle='--', label=f'k={k_90}')
plt.scatter([k_90], [cumulative_ratio[k_90 - 1]], color='green', zorder=3)

plt.xlabel('主成分数量 k')
plt.ylabel('累计方差解释率')
plt.title('Olivetti人脸：主成分数量与累计方差解释率')
plt.xlim(1, 400)
plt.ylim(0, 1.02)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

os.makedirs('./output', exist_ok=True)
plt.savefig('./output/2024150040邓瑞霖Olivetti主成分数量与累计方差解释率.png', dpi=300)
plt.show()

