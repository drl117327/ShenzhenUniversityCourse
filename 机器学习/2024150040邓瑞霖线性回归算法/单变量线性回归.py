import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('./output', exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # mac 常用中文字体
plt.rcParams['axes.unicode_minus'] = False 

data = pd.read_csv('./数据集/regress_data1.csv')
x = data['人口'].values
y = data['收益'].values

# 构建设计矩阵 X: [1, x]
X = np.column_stack([np.ones(x.shape[0]), x])
m = X.shape[0]
# 超参数与初始化
epoch = 1000
alpha = 0.01
theta = np.zeros(2)  # [theta0, theta1]

# 批量梯度下降（向量化）
for _ in range(epoch):
	y_pred = X @ theta
	error = y_pred - y
	grad = (X.T @ error) / m
	theta = theta - alpha * grad

# 最终损失
final_error = X @ theta - y
final_loss = np.sum(final_error ** 2) / (2 * m)

print(f'优化结束时损失值 J(theta): {final_loss:.6f}')
print(f'模型参数 theta0(截距): {theta[0]:.6f}, theta1(斜率): {theta[1]:.6f}')

# 可视化：原始散点 + 拟合直线
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='red', label='原始数据点')

x_line = np.linspace(x.min(), x.max(), 200)
y_line = theta[0] + theta[1] * x_line
plt.plot(x_line, y_line, color='blue', linewidth=2, label='批量梯度下降拟合直线')

plt.xlabel('人口')
plt.ylabel('收益')
plt.title('人口与收益：原始数据与线性回归拟合结果')
plt.legend()
plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖-批量梯度下降拟合结果.png')
plt.show()