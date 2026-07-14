import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

matplotlib.rcParams['font.sans-serif'] = ['Heiti TC']  # mac 常用中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 

olivetti = fetch_olivetti_faces(shuffle=True, random_state=42, data_home="./数据集")
X = olivetti.data
images = olivetti.images
y = olivetti.target

print('数据形状:', X.shape)          # (400, 4096)
print('图像形状:', images.shape)      # (400, 64, 64)
print('标签形状:', y.shape)          # (400,)


# PCA降维：4096 -> 2
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
print('降维后形状:', X_2d.shape)
print('前两主成分方差解释率:', pca.explained_variance_ratio_)
print('前两主成分累计解释率:', np.sum(pca.explained_variance_ratio_))

# 可选：仅选前5个人（标签0~4）着色
selected_labels = np.arange(5)
mask = np.isin(y, selected_labels)
X_plot = X_2d[mask]
y_plot = y[mask]

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for label, color in zip(selected_labels, colors):
	cls_mask = (y_plot == label)
	plt.scatter(
		X_plot[cls_mask, 0],
		X_plot[cls_mask, 1],
		s=35,
		c=color,
		alpha=0.8,
		label=f'person {label}'
	)

plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('Olivetti人脸 PCA降维到2D（前5人）')
plt.legend(title='标签')
plt.grid(alpha=0.2)
plt.tight_layout()

os.makedirs('./output', exist_ok=True)
plt.savefig('./output/2024150040邓瑞霖Olivetti前五个人脸PCA降维到2D.png', dpi=300)
plt.show()
