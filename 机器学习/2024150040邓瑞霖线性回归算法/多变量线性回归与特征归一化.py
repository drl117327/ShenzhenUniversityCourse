import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('./output', exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # mac 常用中文字体
plt.rcParams['axes.unicode_minus'] = False 

data = pd.read_csv('./数据集/regress_data2.csv')
x1 = data['面积'].values
x2 = data['房间数'].values
y = data['价格'].values

# 数据特征标准化
x1_mean, x1_std = x1.mean(), x1.std()
x2_mean, x2_std = x2.mean(), x2.std()
x1 = (x1 - x1_mean) / x1_std
x2 = (x2 - x2_mean) / x2_std
# 构建设计矩阵 X: [1, x1, x2]
X = np.column_stack([np.ones(len(y)), x1, x2])
m, n = X.shape

# 超参数与初始化
epoch = 1000
alpha = 0.01
theta = np.zeros(n)  # 参数初始化为0

# 记录每一轮的损失
loss_history = []

# 批量梯度下降（BGD）
for _ in range(epoch):
	y_pred = X @ theta
	error = y_pred - y
	loss = np.sum(error ** 2) / (2 * m)
	loss_history.append(loss)

	grad = (X.T @ error) / m
	theta = theta - alpha * grad

# 训练结束后的最终损失
final_error = X @ theta - y
final_loss = np.sum(final_error ** 2) / (2 * m)
print(f'优化结束时损失值 J(theta): {final_loss}')
print('最终参数 theta:')
print(f'  theta0(截距): {theta[0]}')
print(f'  theta1(面积系数-标准化后): {theta[1]}')
print(f'  theta2(房间数系数-标准化后): {theta[2]}')

# 绘制 Loss 随 epoch 下降曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch + 1), loss_history, color='blue', linewidth=2)
plt.xlabel('迭代轮数 epoch')
plt.ylabel('训练损失 Loss')
plt.title('BGD训练误差随迭代轮数变化曲线')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./output/2024150040邓瑞霖-BGD多变量线性回归Loss曲线.png')
# plt.show()