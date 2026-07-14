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

# 任务C：(1) 设定主成分数为150并拟合
pca = PCA(n_components=150)
pca.fit(X)

# (2) 获取 components_ 属性
components = pca.components_  # (150, 4096)
print('components_ 形状:', components.shape)

# (3) 提取前5个主成分并 reshape 为 64x64
top_k = 5
eigenfaces = components[:top_k].reshape(top_k, 64, 64)

# (4) 绘制特征脸
fig, axes = plt.subplots(1, top_k, figsize=(14, 3.2))
for i in range(top_k):
	axes[i].imshow(eigenfaces[i], cmap='gray')
	axes[i].set_title(f'特征脸 {i + 1}')
	axes[i].axis('off')

plt.suptitle('前5个PCA特征脸（Eigenfaces）', fontsize=14)
plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖Olivetti前5个PCA特征脸.png', dpi=300)
plt.show()


