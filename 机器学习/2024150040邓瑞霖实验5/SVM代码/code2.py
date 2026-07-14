import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

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
# 2. 优化前基线模型（默认参数）
# ============================================================
model_default = SVC(kernel="rbf", random_state=42)
model_default.fit(x_train_scaled, y_train)
y_pred_default = model_default.predict(x_test_scaled)
acc_default = accuracy_score(y_test, y_pred_default)

print("=" * 60)
print("优化前（默认参数 C=1.0, gamma='scale'）")
print("=" * 60)
print(f"Test Accuracy: {acc_default:.4f}")
print(classification_report(y_test, y_pred_default,
      target_names=cancer.target_names))

# ============================================================
# 3. 定义参数网格 + 执行网格搜索
# ============================================================
param_grid = {
    "C":     [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1],
}

grid_search = GridSearchCV(
    SVC(kernel="rbf", random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True,
)
grid_search.fit(x_train_scaled, y_train)

print("=" * 60)
print("网格搜索结果")
print("=" * 60)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

# ============================================================
# 4. 优化后模型评估
# ============================================================
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(x_test_scaled)
acc_best = accuracy_score(y_test, y_pred_best)

print("=" * 60)
print("优化后（最优参数）")
print("=" * 60)
print(f"Test Accuracy: {acc_best:.4f}")
print(classification_report(y_test, y_pred_best,
      target_names=cancer.target_names))

# ============================================================
# 5. 结果对比
# ============================================================
print("=" * 60)
print("结果对比")
print("=" * 60)
print(f"优化前 Accuracy: {acc_default:.4f}")
print(f"优化后 Accuracy: {acc_best:.4f}")
print(f"提升: {acc_best - acc_default:+.4f}")

# ============================================================
# 6. 可视化1: 参数网格热力图
# ============================================================
results = pd.DataFrame(grid_search.cv_results_)
heatmap_data = results.pivot_table(
    index="param_C", columns="param_gamma",
    values="mean_test_score")

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(heatmap_data.columns)
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)
ax.set_xlabel("gamma")
ax.set_ylabel("C")
ax.set_title("Grid Search: Mean Cross-Validation Accuracy")

for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        text = ax.text(j, i, f"{heatmap_data.iloc[i, j]:.4f}",
                       ha="center", va="center", color="black", fontsize=9)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("grid_search_heatmap.png", dpi=150)
plt.close()

# ============================================================
# 7. 可视化2: 优化前后混淆矩阵对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_, y_pred, title in [
    (axes[0], y_pred_default, f"Before (C=1.0, gamma=scale)\nAcc={acc_default:.4f}"),
    (axes[1], y_pred_best,    f"After (C={grid_search.best_params_['C']}, "
                              f"gamma={grid_search.best_params_['gamma']})\n"
                              f"Acc={acc_best:.4f}"),
]:
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=cancer.target_names).plot(ax=ax_)
    ax_.set_title(title)

plt.tight_layout()
plt.savefig("confusion_matrix_comparison.png", dpi=150)
plt.close()
print("\n图表已保存: grid_search_heatmap.png, confusion_matrix_comparison.png")
