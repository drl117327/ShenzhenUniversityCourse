import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免 headless 环境报错
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

# ============================================================
# 1. 数据加载与预处理
# ============================================================
cancer = load_breast_cancer()
x, y = cancer.data, cancer.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ============================================================
# 2. linear / rbf 核训练与评估
# ============================================================
for kernel in ("linear", "rbf"):
    model = SVC(kernel=kernel, random_state=42)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    print(f"=== SVM kernel={kernel} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# ============================================================
# 3. 网格搜索（RBF 核参数优化）
# ============================================================
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
}

grid_search = GridSearchCV(
    SVC(kernel="rbf", random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(x_train_scaled, y_train)

print("=== GridSearchCV Best Result ===")
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(x_test_scaled)
print(f"Test accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(classification_report(y_test, y_pred_best, target_names=cancer.target_names))

# ============================================================
# 4. 决策边界可视化（PCA 降维 + RBF）
# ============================================================
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

# 用网格搜索最优参数重新训练
svm_rbf = SVC(kernel="rbf", C=grid_search.best_params_["C"],
              gamma=grid_search.best_params_["gamma"], random_state=42)
svm_rbf.fit(x_train_pca, y_train)

# 绘制决策边界
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
                          edgecolors="k", s=50)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.tight_layout()
    plt.savefig("decision_boundary_rbf.png", dpi=150)
    plt.close()

y_pred_pca = svm_rbf.predict(x_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

plot_decision_boundary(
    svm_rbf, x_train_pca, y_train,
    f"SVM RBF Decision Boundary (PCA 2D)\n"
    f"C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}\n"
    f"Test Accuracy = {acc_pca:.4f}"
)
