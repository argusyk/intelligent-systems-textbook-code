# Приватність: атака на членство, механізм Лапласа (диференційна приватність),
# федеративне навчання трьох авіакомпаній (усереднення моделей, FedAvg)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

rng = np.random.default_rng(1)
def make(n, shift=0.0):
    fault = rng.random(n) < 0.3
    X = np.column_stack([rng.normal(610 + shift, 25, n) + fault * rng.normal(45, 25, n),
                         rng.normal(36, 6, n) + fault * rng.normal(8, 6, n),
                         rng.normal(118, 8, n) + fault * rng.normal(14, 8, n)])
    return (X - [620, 38, 122]) / [30, 7, 10], fault.astype(int)

# 1. Атака на членство: чи був запис у навчальній вибірці?
X_in, y_in = make(300); X_out, y_out = make(300)
for name, m in [("перенавчений ліс", RandomForestClassifier(200, random_state=0)),
                ("регуляризований ліс", RandomForestClassifier(200, min_samples_leaf=20, random_state=0))]:
    m.fit(X_in, y_in)
    conf = lambda X, y: m.predict_proba(X)[np.arange(len(y)), y]   # упевненість у правильному класі
    c_in, c_out = conf(X_in, y_in), conf(X_out, y_out)
    auc = roc_auc_score(np.r_[np.ones(300), np.zeros(300)], np.r_[c_in, c_out])
    print(f"{name:20s} упевненість: у вибірці {c_in.mean():.3f}, поза нею {c_out.mean():.3f}; "
          f"AUC атаки = {auc:.3f}")

# 2. Механізм Лапласа: середня EGT флоту без розкриття окремої установки
egt = rng.normal(640, 30, 200).clip(500, 800)
sens = (800 - 500) / len(egt)                                      # чутливість середнього
for eps in [0.1, 0.5, 1.0, 5.0]:
    noisy = egt.mean() + rng.laplace(0, sens / eps, 1000)
    print(f"eps = {eps:3.1f}: середня похибка опублікованого значення {np.abs(noisy - egt.mean()).mean():.2f} °C")

# 3. Федеративне навчання: три авіакомпанії, дані не залишають компанію
parts = [make(n, s) for n, s in [(30, -15), (400, 0), (40, 20)]]
X_test, y_test = make(3000, 5)
def local_step(w, b, X, y, lr=0.5, epochs=5):
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w, b = w - lr * X.T @ (p - y) / len(y), b - lr * (p - y).mean()
    return w, b
w, b = np.zeros(3), 0.0
for rnd in range(40):                                               # раунди FedAvg
    upd = [local_step(w, b, X, y) for X, y in parts]
    n = np.array([len(y) for _, y in parts]); a = n / n.sum()
    w = sum(ai * u[0] for ai, u in zip(a, upd)); b = sum(ai * u[1] for ai, u in zip(a, upd))
f1_fed = f1_score(y_test, (X_test @ w + b > 0).astype(int))
f1_local = [f1_score(y_test, LogisticRegression().fit(X, y).predict(X_test)) for X, y in parts]
Xc = np.vstack([X for X, _ in parts]); yc = np.r_[tuple(y for _, y in parts)]
f1_central = f1_score(y_test, LogisticRegression().fit(Xc, yc).predict(X_test))
print("F1 лише на власних даних:", [round(v, 3) for v in f1_local])
print(f"F1 федеративна модель: {f1_fed:.3f}; F1 централізована (усі дані разом): {f1_central:.3f}")
