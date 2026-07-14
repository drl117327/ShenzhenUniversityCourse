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

# 任务B：选取第1张人脸（索引0）作为测试样本
test_idx = 0
test_face = X[test_idx:test_idx + 1]  # shape: (1, 4096)

# 分别使用不同主成分进行重构
n_components_list = [10, 50, 150, 300]
reconstructed_faces = []

for k in n_components_list:
	pca = PCA(n_components=k, svd_solver='randomized', whiten=False, random_state=42)
	pca.fit(X)
	test_face_low = pca.transform(test_face)
	test_face_recon = pca.inverse_transform(test_face_low)
	reconstructed_faces.append(test_face_recon.reshape(64, 64))

# 可视化：原图 + 4张重构图
fig, axes = plt.subplots(1, 5, figsize=(15, 4))

axes[0].imshow(images[test_idx], cmap='gray')
axes[0].set_title('原始图像\n(index=0)')
axes[0].axis('off')

for i, (k, recon_img) in enumerate(zip(n_components_list, reconstructed_faces), start=1):
	axes[i].imshow(recon_img, cmap='gray')
	axes[i].set_title(f'重构图像\nk={k}')
	axes[i].axis('off')

plt.suptitle('Olivetti人脸PCA重构对比（64×64）', fontsize=14)
plt.tight_layout()

os.makedirs('./output', exist_ok=True)
plt.savefig('./output/2024150040邓瑞霖Olivetti人脸PCA重构对比.png', dpi=300)
plt.show()

