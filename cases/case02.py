# Перцептрон Розенблатта на реальних даних: ірис щетинистий проти різнобарвного
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
mask = y < 2                                  # два класи з трьох
X, t = X[mask][:, [0, 2]], y[mask]            # довжина чашолистка і пелюстки
w, b = np.zeros(2), 0.0
for epoch in range(1, 21):
    errors = 0
    for xi, ti in zip(X, t):
        yi = 1 if xi @ w + b > 0 else 0
        if yi != ti:                          # правило (14.3) / розділ 2
            w += 0.1 * (ti - yi) * xi
            b += 0.1 * (ti - yi)
            errors += 1
    if errors == 0:
        break
print(f"збіжність на епосі {epoch}; ваги {w.round(2)}, зсув {b:.2f}")
acc = np.mean([(1 if xi @ w + b > 0 else 0) == ti for xi, ti in zip(X, t)])
print(f"точність на 100 квітках: {acc:.2f}")
