import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # mac 常用中文字体
plt.rcParams['axes.unicode_minus'] = False 

# 生成测试数据
X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mu_0 = X_train[y_train == 0].mean(axis=0)
mu_1 = X_train[y_train == 1].mean(axis=0)
mu_2 = X_train[y_train == 2].mean(axis=0)
print('3个类别样本在各特征上的均值向量:', mu_0, mu_1, mu_2)
# 计算训练集中所有样本的总体均值
mu = X_train.mean(axis=0)
print('训练集中所有样本的总体均值:', mu)
# 每个类别的先验概率
p_0 = (y_train == 0).sum() / len(y_train)
p_1 = (y_train == 1).sum() / len(y_train)
p_2 = (y_train == 2).sum() / len(y_train)
print('3个类别在训练集中的先验概率:', p_0, p_1, p_2)

# 计算散度矩阵
# 计算类内散度矩阵
S_w = np.zeros((X_train.shape[1], X_train.shape[1]))
for c in np.unique(y_train):
    X_c = X_train[y_train == c]
    S_w += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1) / (X_c.shape[0])
print('类内散度矩阵 S_w:\n', S_w)
# 计算类间散度矩阵
S_b = np.zeros((X_train.shape[1], X_train.shape[1]))
for c in np.unique(y_train):
    n_c = (y_train == c).sum()
    mu_c = X_train[y_train == c].mean(axis=0)
    S_b += n_c * np.outer(mu_c - mu, mu_c - mu)
print('类间散度矩阵 S_b:\n', S_b)

# 计算投影空间与决策函数
S_w_inv = np.linalg.inv(S_w)
eigenvalues, eigenvectors = np.linalg.eig(S_w_inv @ S_b)
# 按照特征值从大到小排序
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
print('特征值:', eigenvalues)
print('特征向量:\n', eigenvectors)

# 选择前 k 个特征向量构成投影矩阵 W
k = 2
W = eigenvectors[:, :k]
print('投影矩阵 W:\n', W)
# 将训练集各类均值向量乘以投影矩阵W
mu_0_proj = W.T @ mu_0
mu_1_proj = W.T @ mu_1
mu_2_proj = W.T @ mu_2
print('3个类别均值向量在投影空间的坐标:', mu_0_proj, mu_1_proj, mu_2_proj)

# 进行测试集预测与评估
X_test_proj = X_test @ W
y_pred = []
for x in X_test_proj:
    d_0 = np.linalg.norm(x - mu_0_proj)
    d_1 = np.linalg.norm(x - mu_1_proj)
    d_2 = np.linalg.norm(x - mu_2_proj)
    if d_0 < d_1 and d_0 < d_2:
        y_pred.append(0)
    elif d_1 < d_0 and d_1 < d_2:
        y_pred.append(1)
    else:
        y_pred.append(2)
accuracy = accuracy_score(y_test, y_pred)
print('测试集预测准确率:', accuracy)