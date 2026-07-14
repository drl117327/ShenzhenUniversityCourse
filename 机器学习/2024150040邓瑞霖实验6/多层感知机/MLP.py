"""MLP 数据准备：Iris 加载、标准化、训练/测试划分。"""

from __future__ import annotations
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

os.makedirs("./output", exist_ok=True)
def prepare_iris_data(test_size: float = 0.2, random_state: int = 42):

	iris = load_iris()
	X = iris.data
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
	X_train, X_test, y_train, y_test, scaler = prepare_iris_data()
	print("训练集大小:", X_train.shape, "测试集大小:", X_test.shape)
    
	model = MLPClassifier(
		hidden_layer_sizes=(10),
		activation="relu",
		solver="adam",
		random_state=42,
		max_iter=500,
	)
	model.fit(X_train, y_train)
	loss_curve = model.loss_curve_

	plt.figure(figsize=(6, 4))
	plt.plot(range(1, len(loss_curve) + 1), loss_curve)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("MLP Training Loss Curve")
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.tight_layout()
	plt.savefig("./output/2024150040邓瑞霖MLP_loss.png")

	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	cm = confusion_matrix(y_test, y_pred)
	print("最终的loss:", loss_curve[-1])
	print("迭代次数:", model.n_iter_)
	print("测试集准确率:", f"{acc:.4f}")
	print("混淆矩阵:\n", cm)
