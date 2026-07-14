import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # mac 常用中文字体
plt.rcParams['axes.unicode_minus'] = False 

X_dummy, y_dummy = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, 
                                       n_clusters_per_class=1, class_sep=1.5, random_state=42)

mu_0 = X_dummy[y_dummy == 0].mean(axis=0)
mu_1 = X_dummy[y_dummy == 1].mean(axis=0)
mu = X_dummy.mean(axis=0)
print('类别0的均值向量:', mu_0)
print('类别1的均值向量:', mu_1)
print('总体均值向量:', mu)

# 计算散度矩阵
S_w = np.zeros((X_dummy.shape[1], X_dummy.shape[1]))
for c in np.unique(y_dummy):
    X_c = X_dummy[y_dummy == c]
    S_w += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1) / (X_c.shape[0])
print('类内散度矩阵 S_w:\n', S_w)
S_b = ((mu_0 - mu_1).reshape(-1, 1)) @ (mu_0 - mu_1).reshape(1, -1)
print('类间散度矩阵 S_b:\n', S_b)

# 特征值分解与投影
S_w_inv = np.linalg.inv(S_w)
eigenvalues, eigenvectors = np.linalg.eig(S_w_inv @ S_b)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sorted_indices]
print('特征值:', eigenvalues)
print('特征向量:\n', eigenvectors)
k = 1
W = eigenvectors[:, :k]
print('投影矩阵 W:\n', W)
# 投影坐标
Y = X_dummy @ W
print('投影后的坐标:\n', Y)
# 可视化投影结果
plt.figure(figsize=(8, 6))
Y_1d = Y.ravel()
Y_zero = np.zeros_like(Y_1d)

for c in np.unique(y_dummy):
    mask = y_dummy == c
    plt.scatter(
        Y_1d[mask],
        Y_zero[mask],
        label=f'类别 {c}',
        alpha=0.8,
        s=35
    )
os.makedirs('./output', exist_ok=True)
plt.xlabel('投影值 Y')
plt.ylabel('0')
plt.title('手写 LDA 的一维投影散点图')
plt.yticks([0])
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖LDA降维.png', dpi=300)
plt.show()