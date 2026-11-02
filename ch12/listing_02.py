# ANFIS: нечітка система Сугено першого порядку, параметри якої навчаються за даними.
# Гібридне навчання: висновки правил – методом найменших квадратів, функції належності –
# градієнтним спуском (градієнт обчислюється чисельно, щоб код залишався коротким).
import numpy as np

rng = np.random.default_rng(0)
n = 400
egt = rng.uniform(560, 740, n); ts = rng.uniform(25, 65, n)
z = 0.06 * (egt - 670) + 0.12 * (ts - 48) + 0.002 * (egt - 670) * (ts - 48)
risk = 100 / (1 + np.exp(-z))
y = risk + rng.normal(0, 3, n)                                     # «виміряний» ризик, %
X = np.column_stack([(egt - 650) / 50, (ts - 45) / 12])            # нормовані входи
tr, te = np.arange(n) < 300, np.arange(n) >= 300

def firing(X, P):
    """Шари 1-3: гаусові функції належності, добуток (І), нормування ступенів активізації."""
    c, s = P[:6].reshape(2, 3), np.abs(P[6:].reshape(2, 3)) + 1e-3
    mu = [np.exp(-((X[:, [i]] - c[i]) / s[i]) ** 2) for i in range(2)]   # 3 терми на вхід
    w = (mu[0][:, :, None] * mu[1][:, None, :]).reshape(len(X), 9)      # 9 правил
    return w / w.sum(1, keepdims=True)

def design(X, P):
    wn = firing(X, P)  # шар 4: wn_k * (a0 + a1 x1 + a2 x2)
    Xa = np.column_stack([np.ones(len(X)), X])
    return (wn[:, :, None] * Xa[:, None, :]).reshape(len(X), 27)

def fit_consequents(X, y, P):
    A = design(X, P)
    return np.linalg.lstsq(A, y, rcond=None)[0]

def rmse(X, y, P, theta):
    return float(np.sqrt(np.mean((design(X, P) @ theta - y) ** 2)))

Xl = np.column_stack([np.ones(n), X])                               # еталон: лінійна регресія
beta = np.linalg.lstsq(Xl[tr], y[tr], rcond=None)[0]
rmse_lin = np.sqrt(np.mean((Xl[te] @ beta - y[te]) ** 2))
print(f"лінійна регресія:       тест RMSE {rmse_lin:.2f}")
P = np.r_[[-1.2, 0, 1.2], [-1.2, 0, 1.2], [0.8] * 6]  # експертні терми «низька/норма/висока»
theta = fit_consequents(X[tr], y[tr], P)
print(f"лише висновки (МНК):    RMSE навч. {rmse(X[tr], y[tr], P, theta):.2f}, тест "
      f"{rmse(X[te], y[te], P, theta):.2f}")
lr, h = 0.02, 1e-4
for epoch in range(150):                                           # гібридний цикл
    theta = fit_consequents(X[tr], y[tr], P)
    base = rmse(X[tr], y[tr], P, theta) ** 2
    g = np.zeros_like(P)
    for j in range(len(P)):
        Pj = P.copy(); Pj[j] += h
        g[j] = (rmse(X[tr], y[tr], Pj, theta) ** 2 - base) / h
    P -= lr * g
theta = fit_consequents(X[tr], y[tr], P)
print(f"після налаштування ФН: RMSE навч. {rmse(X[tr], y[tr], P, theta):.2f}, тест "
      f"{rmse(X[te], y[te], P, theta):.2f}")
c = P[:6].reshape(2, 3)
print("центри термів EGT, °C:", (650 + 50 * c[0]).round(0), " час запуску, с:",
      (45 + 12 * c[1]).round(1))
k = 8  # правило «EGT висока І запуск затяжний»
a0, a1, a2 = theta[3 * k:3 * k + 3]
print(f"правило 9: ЯКЩО EGT висока І запуск затяжний, ТО ризик = {a0:.1f} {a1:+.1f}*x1 "
      f"{a2:+.1f}*x2")
