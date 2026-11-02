# Rare faults: ROC and PR curves, choosing the threshold by error cost, calibration
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                             precision_score, accuracy_score, brier_score_loss)

rng = np.random.default_rng(5)
n = 6000
fault = rng.random(n) < 0.04                                   # 4 % faulty
X = np.column_stack([rng.normal(610, 25, n) + fault * rng.normal(40, 20, n),
                     rng.normal(36, 6, n) + fault * rng.normal(7, 6, n),
                     rng.normal(118, 8, n) + fault * rng.normal(10, 8, n)])
X = (X - X.mean(0)) / X.std(0)
y = fault.astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

print(f"always 'healthy': accuracy = {1 - y_te.mean():.3f}, recall = 0")
for name, cw in [("ordinary", None), ("class-weighted", "balanced")]:
    m = LogisticRegression(class_weight=cw).fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]; pred = (p >= 0.5).astype(int)
    print(f"{name:14s} acc={accuracy_score(y_te, pred):.3f} "
          f"recall={recall_score(y_te, pred):.3f} "
          f"precision={precision_score(y_te, pred):.3f} ROC AUC={roc_auc_score(y_te, p):.3f} "
          f"PR AUC={average_precision_score(y_te, p):.3f} "
          f"Brier={brier_score_loss(y_te, p):.4f}")

m = LogisticRegression().fit(X_tr, y_tr)
p = m.predict_proba(X_te)[:, 1]
C_FN, C_FP = 50, 1                                             # a miss is 50 times more expensive
th = np.linspace(0.01, 0.9, 90)
cost = [C_FN * np.sum((p < t) & (y_te == 1)) + C_FP * np.sum((p >= t) & (y_te == 0))
        for t in th]
t_best = th[int(np.argmin(cost))]
print(f"cost-based threshold: {t_best:.2f}; cost {min(cost)} vs {cost[49]} at threshold 0.5")
pred = p >= t_best
print(f"at threshold {t_best:.2f}: recall={recall_score(y_te, pred):.3f} "
      f"precision={precision_score(y_te, pred):.3f}")
