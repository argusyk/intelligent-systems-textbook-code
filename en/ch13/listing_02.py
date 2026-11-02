import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(42)
n = 600
fault = rng.random(n) < 0.3                                    # 30 % fuel system faults
egt   = rng.normal(610, 25, n) + fault * rng.normal(55, 20, n)
start = rng.normal(36, 6, n)  + fault * rng.normal(10, 6, n)
fuel  = rng.normal(118, 8, n) + fault * rng.normal(18, 8, n)
vib   = rng.normal(3.0, 0.8, n)                                # uninformative feature
X, y = np.column_stack([egt, start, fuel, vib]), fault.astype(int)

models = {"logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
          "random forest": RandomForestClassifier(n_estimators=200, random_state=0)}
for name, m in models.items():
    f1 = cross_val_score(m, X, y, cv=5, scoring="f1")
    print(f"{name:20s} F1 = {f1.mean():.3f} +- {f1.std():.3f}")
