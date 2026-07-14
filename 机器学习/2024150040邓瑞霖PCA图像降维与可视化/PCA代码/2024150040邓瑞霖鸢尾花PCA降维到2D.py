import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from PIL import Image
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Heiti TC']  # mac 常用中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 

# 加载与预处理
iris = load_iris()

data = pd.read_csv('./数据集/iris.csv')
X = data.iloc[:, 1: 5].values
y = data['Species'].values
feature_names = data.columns[1:5].tolist()

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mean_before = X.mean(axis=0)
std_before = X.std(axis=0)
mean_after = X_scaled.mean(axis=0)
std_after = X_scaled.std(axis=0)

# 降低维度 4 -> 2
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
print(X_reduced)
print('PCA(2) explained_variance_ratio_:', pca.explained_variance_ratio_)

# 数据重构（逆变换）: 2维 -> 4维(标准化空间) -> 4维(原始尺度)
X_reconstructed_scaled = pca.inverse_transform(X_reduced)
X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)

# 计算重构误差（MSE）
mse = np.mean((X - X_reconstructed) ** 2)
print('原始数据与重构数据的均方误差 MSE:', mse)

plt.figure(figsize=(7, 5))
Y = set(y)
color_map = {label: color for label, color in zip(Y, ['red', 'green', 'blue'])}
for label in sorted(Y):
    mask = (y == label)
    plt.scatter(
        X_reduced[mask, 0],
        X_reduced[mask, 1],
        c=color_map[label],
        label=label,
        alpha=0.8
    )
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend(title='类别')
plt.title('PCA降维到2D散点图')
plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖PCA降维到2D散点图.png', dpi=300)
plt.show()

# 对比可视化：原始数据前两个特征 vs PCA降维后二维数据
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for label in sorted(Y):
    mask = (y == label)
    axes[0].scatter(
        X[mask, 0],
        X[mask, 1],
        c=color_map[label],
        label=label,
        alpha=0.8
    )
axes[0].set_xlabel(feature_names[0])
axes[0].set_ylabel(feature_names[1])
axes[0].set_title('原始数据（前两个特征）')
axes[0].legend(title='类别')

for label in sorted(Y):
    mask = (y == label)
    axes[1].scatter(
        X_reduced[mask, 0],
        X_reduced[mask, 1],
        c=color_map[label],
        label=label,
        alpha=0.8
    )
axes[1].set_xlabel('主成分1')
axes[1].set_ylabel('主成分2')
axes[1].set_title('PCA降维后二维数据')
axes[1].legend(title='类别')

plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖PCA原始_vs_降维.png', dpi=300)
plt.show()

loadings = pca.components_

print('PCA(2)载荷矩阵 pca.components_:')
print(loadings)

# 分析每个主成分贡献最大的原始特征（按绝对值）
for i in range(loadings.shape[0]):
    abs_weights = np.abs(loadings[i])
    top_idx = int(np.argmax(abs_weights))
    print(
        f'PC{i + 1}贡献最大的特征: {feature_names[top_idx]} '
        f'(载荷={loadings[i, top_idx]:.4f}, |载荷|={abs_weights[top_idx]:.4f})'
    )

# 热力图可视化
plt.figure(figsize=(8, 4))
im = plt.imshow(loadings, cmap='coolwarm', aspect='auto')
plt.colorbar(im, label='载荷权重')
plt.xticks(np.arange(len(feature_names)), feature_names, rotation=30)
plt.yticks(np.arange(2), ['PC1', 'PC2'])
plt.title('PCA主成分载荷图（Heatmap）')

for r in range(loadings.shape[0]):
    for c in range(loadings.shape[1]):
        plt.text(c, r, f'{loadings[r, c]:.2f}', ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖PCA载荷图.png', dpi=300)
plt.show()
