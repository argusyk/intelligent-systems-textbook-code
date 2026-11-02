# Порівняння моделей на даних з нелінійною межею: kNN, SVM, ліс, градієнтний бустинг
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate

rng = np.random.default_rng(1)
n = 1500
egt, fuel = rng.normal(0, 1, n), rng.normal(0, 1, n)          # стандартизовані EGT і витрата
start, vib, oil = rng.normal(0, 1, (3, n))
# несправність: EGT і витрата «розійшлися» (висока EGT при низькій витраті або навпаки)
score = np.abs(egt - fuel) + 0.6 * start + rng.normal(0, 0.35, n)
y = (score > 1.6).astype(int)
X = np.column_stack([egt, fuel, start, vib, oil])
print(f"частка несправних: {y.mean():.2f}")
models = {
    "логістична регресія": make_pipeline(StandardScaler(), LogisticRegression()),
    "kNN, k = 15": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "SVM, RBF-ядро": make_pipeline(StandardScaler(), SVC(C=3, gamma="scale")),
    "випадковий ліс": RandomForestClassifier(300, random_state=0),
    "градієнтний бустинг": HistGradientBoostingClassifier(random_state=0)}
for name, m in models.items():
    r = cross_validate(m, X, y, cv=5, scoring=["f1", "roc_auc"])
    print(f"{name:20s} F1 = {r['test_f1'].mean():.3f}  ROC AUC = "
          f"{r['test_roc_auc'].mean():.3f}  "
          f"час = {r['fit_time'].mean():.2f} с")
X2 = np.column_stack([X, np.abs(egt - fuel)])                  # ознака, підказана фізикою
r = cross_validate(models["логістична регресія"], X2, y, cv=5, scoring=["f1", "roc_auc"])
print(f"{'логістична + |EGT-в.|':20s} F1 = {r['test_f1'].mean():.3f}  "
      f"ROC AUC = {r['test_roc_auc'].mean():.3f}")
