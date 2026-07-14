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
mean_before = X.mean(axis=0)
std_before = X.std(axis=0)
X_scaled= (X - mean_before) / std_before

mean_after = X_scaled.mean(axis=0)
std_after = X_scaled.std(axis=0)

print(X)
print(y)
print(X_scaled)
print('标准化前 X 的均值:', mean_before)
print('标准化后 X 的均值:', mean_after)
print('标准化后 X 的标准差:', std_after)


# 不指定降维维度，提取所有主成分
pca = PCA()
pca.fit(X_scaled)

# 所有主成分方差贡献率与累计贡献率
explained_ratio = pca.explained_variance_ratio_
cumulative_ratio = np.cumsum(explained_ratio)
eigenvalues = pca.explained_variance_

print('所有主成分的方差贡献率:', explained_ratio)
print('所有主成分特征值:', eigenvalues)
for i, ratio in enumerate(explained_ratio, start=1):
    print(f'第{i}主成分方差贡献率: {ratio:.6f}')

print('所有主成分的累计方差贡献率:', cumulative_ratio)


# 报告分析：前几个主成分保留的信息
max_k = min(3, len(cumulative_ratio))
for k in range(1, max_k + 1):
    print(f'前{k}个主成分累计保留信息: {cumulative_ratio[k - 1]:.2%}')

threshold = 0.95
k_95 = int(np.argmax(cumulative_ratio >= threshold) + 1)
if cumulative_ratio[-1] >= threshold:
    print(f'达到95%信息保留率至少需要前{k_95}个主成分。')
else:
    print('所有主成分累计信息保留率不足95%，请检查数据或阈值设置。')

# 绘制方差贡献率图与累计方差贡献率图
components = np.arange(1, len(explained_ratio) + 1)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(components, explained_ratio, color='skyblue', edgecolor='black')
plt.plot(components, explained_ratio, marker='o', color='royalblue')
plt.xlabel('主成分编号')
plt.ylabel('方差贡献率')
plt.title('PCA方差贡献率图')
plt.xticks(components)

plt.subplot(1, 2, 2)
plt.plot(components, cumulative_ratio, marker='o', color='orange')
plt.axhline(y=0.95, color='red', linestyle='--', label='95%阈值')
plt.xlabel('主成分编号')
plt.ylabel('累计方差贡献率')
plt.title('PCA累计方差贡献率图')
plt.xticks(components)
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()
os.makedirs('./output', exist_ok=True)
plt.savefig('./output/2024150040邓瑞霖方差贡献率图.png', dpi=300)

# 绘制碎石图（Scree Plot）并寻找手肘位置
plt.figure(figsize=(6, 4))
plt.plot(components, eigenvalues, marker='o', color='teal')
plt.xlabel('主成分编号')
plt.ylabel('特征值 (explained_variance_)')
plt.title('PCA碎石图 (Scree Plot)')
plt.xticks(components)

# 启发式寻找手肘点：相邻特征值下降幅度最大的拐点后一个位置
if len(eigenvalues) > 1:
    diffs = np.diff(eigenvalues)
    elbow = int(np.argmin(diffs) + 2)
    plt.axvline(elbow, color='red', linestyle='--', label=f'手肘位置: PC{elbow}')
    plt.legend()
    print(f'碎石图启发式手肘位置: 第{elbow}个主成分。')
else:
    print('主成分数量不足，无法判断手肘位置。')

plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖碎石图.png', dpi=300)
plt.show()


