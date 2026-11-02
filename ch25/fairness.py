# Аудит справедливості моделі кредитного скорингу та методи зменшення упередженості
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(7)
n = 6000
group = (rng.random(n) < 0.35).astype(int)                 # 1 – група B (наприклад, мешканці сіл), 0 – група A
income = rng.normal(30 - 6 * group, 8, n)                  # історично нижчі доходи групи B
history = rng.normal(6, 2, n)                              # кредитна історія, бали
district = group + rng.normal(0, 0.3, n)                   # «район» – проксі-ознака групи
ability = 0.08 * income + 0.5 * history + rng.normal(0, 1, n)
repaid = (ability > np.median(ability)).astype(int)        # справжня здатність повернути кредит
# історичні рішення з упередженням: групі B частіше відмовляли за однакової здатності
label = ((ability - 0.8 * group + rng.normal(0, 0.3, n)) > np.median(ability)).astype(int)

def report(name, approve, grp, y_true):
    rate = [approve[grp == g].mean() for g in (0, 1)]
    tpr = [approve[(grp == g) & (y_true == 1)].mean() for g in (0, 1)]
    acc = (approve == y_true).mean()
    print(f"{name:34s} схвалено A {rate[0]:.2f} B {rate[1]:.2f} | DI = {rate[1]/rate[0]:.2f} | "
          f"TPR A {tpr[0]:.2f} B {tpr[1]:.2f} | точність {acc:.3f}")

tr, te = np.arange(n) < 4000, np.arange(n) >= 4000
def fit(X, w=None):
    return LogisticRegression(max_iter=1000).fit(X[tr], label[tr], sample_weight=None if w is None else w[tr])

X_full = np.column_stack([income, history, group])
X_blind = np.column_stack([income, history, district])       # без групи, але з проксі «район»
g, y = group, label

m1 = fit(X_full); p1 = m1.predict_proba(X_full)[:, 1]
m2 = fit(X_blind); p2 = m2.predict_proba(X_blind)[:, 1]
# перезважування: вага = P(група) P(мітка) / P(група, мітка)
w = np.ones(n)
for gg in (0, 1):
    for yy in (0, 1):
        m = (g == gg) & (y == yy) & tr
        w[(g == gg) & (y == yy)] = (np.mean(g[tr] == gg) * np.mean(y[tr] == yy)) / m.mean() * tr.mean()
m3 = fit(X_blind, w); p3 = m3.predict_proba(X_blind)[:, 1]
# групові пороги (підбір на навчальній частині), що вирівнюють частку схвалених серед тих, хто поверне кредит (рівні можливості)
thr = {0: 0.5}
tpr_a = (p3[tr & (g == 0) & (repaid == 1)] > 0.5).mean()
cands = np.linspace(0.2, 0.8, 121)
thr[1] = cands[np.argmin([abs((p3[tr & (g == 1) & (repaid == 1)] > t).mean() - tpr_a) for t in cands])]
a4 = np.where(g == 1, p3 > thr[1], p3 > thr[0]).astype(int)

for name, a in [("1) модель з ознакою групи", (p1 > 0.5).astype(int)), ("2) без групи, з проксі «район»", (p2 > 0.5).astype(int)),
                ("3) перезважування", (p3 > 0.5).astype(int)), (f"4) + поріг для B = {thr[1]:.2f}", a4)]:
    report(name, a[te], group[te], repaid[te])
