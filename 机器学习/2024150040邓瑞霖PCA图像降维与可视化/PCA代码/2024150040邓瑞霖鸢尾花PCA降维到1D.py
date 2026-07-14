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
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X_scaled)
Y = set(y)
color_map = {label: color for label, color in zip(Y, ['red', 'green', 'blue'])}
color = [color_map[label] for label in y]

for label in sorted(Y):
    mask = (y == label)
    plt.scatter(
        X_reduced[mask],
        np.zeros_like(X_reduced[mask]),
        c=color_map[label],
        label=label,
        alpha=0.8
    )
plt.xlabel('主成分1')
plt.legend(title='类别')
plt.title('PCA降维到1D散点图')
plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖PCA降维到1D散点图.png', dpi=300)
plt.show()


