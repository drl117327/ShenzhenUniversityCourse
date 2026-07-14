from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # mac 常用中文字体
plt.rcParams['axes.unicode_minus'] = False


def main() -> None:
	# 1) 加载数据并划分训练/测试集
	faces = fetch_olivetti_faces(shuffle=True, random_state=42, data_home="./数据集")
	X = faces.data      # (400, 4096)
	y = faces.target    # 40 类标签（0~39）

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.3,
		random_state=42,
		stratify=y
	)

	print('原始数据形状:')
	print(f'  X: {X.shape}, y: {y.shape}')
	print('训练/测试划分:')
	print(f'  X_train: {X_train.shape}, y_train: {y_train.shape}')
	print(f'  X_test : {X_test.shape}, y_test : {y_test.shape}')

	# 2) LDA 识别与降维（Fisherfaces）
	lda = LinearDiscriminantAnalysis()
	lda.fit(X_train, y_train)

	X_train_lda = lda.transform(X_train)
	X_test_lda = lda.transform(X_test)

	print('\nLDA 降维信息:')
	print(f'  X_train 经过 LDA 后形状: {X_train_lda.shape}')
	print(f'  降维后特征列数: {X_train_lda.shape[1]}')
	print('  理论上限 min(n_features, n_classes - 1) = min(4096, 40 - 1) = 39')

	y_pred_lda = lda.predict(X_test)
	acc_lda = accuracy_score(y_test, y_pred_lda)
	print(f'\nLDA 自带分类器在测试集上的准确率: {acc_lda:.4f}')

	# 3) LDA + SVM
	svm = SVC(kernel='linear', random_state=42)
	svm.fit(X_train_lda, y_train)
	y_pred_svm = svm.predict(X_test_lda)
	acc_svm = accuracy_score(y_test, y_pred_svm)

	print(f'LDA + 线性 SVM 在测试集上的准确率: {acc_svm:.4f}')

	diff = acc_svm - acc_lda
	print(f'两者准确率差值 (SVM - LDA): {diff:+.4f}')

if __name__ == '__main__':
	main()
