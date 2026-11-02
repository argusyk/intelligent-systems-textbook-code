# Comparing models on data with a nonlinear boundary: kNN, SVM, forest, gradient boosting
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
egt, fuel = rng.normal(0, 1, n), rng.normal(0, 1, n)          # standardised EGT and fuel flow
start, vib, oil = rng.normal(0, 1, (3, n))
# fault: EGT and fuel flow "diverged" (high EGT with low fuel flow or vice versa)
score = np.abs(egt - fuel) + 0.6 * start + rng.normal(0, 0.35, n)
y = (score > 1.6).astype(int)
X = np.column_stack([egt, fuel, start, vib, oil])
print(f"share of faulty units: {y.mean():.2f}")
models = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "kNN, k = 15": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "SVM, RBF kernel": make_pipeline(StandardScaler(), SVC(C=3, gamma="scale")),
    "random forest": RandomForestClassifier(300, random_state=0),
    "gradient boosting": HistGradientBoostingClassifier(random_state=0)}
for name, m in models.items():
    r = cross_validate(m, X, y, cv=5, scoring=["f1", "roc_auc"])
    print(f"{name:20s} F1 = {r['test_f1'].mean():.3f}  ROC AUC = "
          f"{r['test_roc_auc'].mean():.3f}  "
          f"time = {r['fit_time'].mean():.2f} s")
X2 = np.column_stack([X, np.abs(egt - fuel)])                  # a feature suggested by physics
r = cross_validate(models["logistic regression"], X2, y, cv=5, scoring=["f1", "roc_auc"])
print(f"{'logistic + |EGT-fuel|':20s} F1 = {r['test_f1'].mean():.3f}  "
      f"ROC AUC = {r['test_roc_auc'].mean():.3f}")
