# Моніторинг моделі кредитного скорингу: індекс стабільності популяції (PSI) щомісяця
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(8)
def month(n, shift=0.0, concept=0.0):
    income = rng.lognormal(10.2 - shift * 0.3, 0.5, n)            # доходи (зсув – криза)
    debt = rng.uniform(0, 1, n) + shift * 0.15                    # боргове навантаження
    age = rng.integers(21, 65, n)
    z = (-2 + 3.0 * debt - 0.8 * (np.log(income) - 10.2) - 0.02 * (age - 40)
         + concept * (2.5 * (age < 30) - 3.0 * debt))
    return np.column_stack([np.log(income), debt,
                            age]), (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

X0, y0 = month(5000)
model = LogisticRegression(max_iter=1000).fit(X0, y0)
ref = model.predict_proba(X0)[:, 1]
edges = np.quantile(ref, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf

def psi(new):
    a = np.histogram(ref, edges)[0] / len(ref); b = np.histogram(new, edges)[0] / len(new)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))

scenario = [(0, 0), (0, 0), (0.3, 0), (0.7, 0), (1.0, 0), (1.0, 1.2)]
for m, (shift, concept) in enumerate(scenario, 1):
    X, y = month(2000, shift, concept)
    p = model.predict_proba(X)[:, 1]
    v = psi(p)
    status = "стабільно" if v < 0.1 else ("увага" if v < 0.25 else "сильний зсув")
    print(f"місяць {m}: PSI = {v:.3f} ({status:12s}) ROC AUC = {roc_auc_score(y, p):.3f}, "
          f"частка дефолтів {y.mean():.2f}")
