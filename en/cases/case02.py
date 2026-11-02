# Rosenblatt's perceptron on real data: Iris setosa versus Iris versicolor
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
mask = y < 2                                  # two classes out of three
X, t = X[mask][:, [0, 2]], y[mask]            # sepal length and petal length
w, b = np.zeros(2), 0.0
for epoch in range(1, 21):
    errors = 0
    for xi, ti in zip(X, t):
        yi = 1 if xi @ w + b > 0 else 0
        if yi != ti:                          # rule (14.3) / Chapter 2
            w += 0.1 * (ti - yi) * xi
            b += 0.1 * (ti - yi)
            errors += 1
    if errors == 0:
        break
status = f"converged at epoch {epoch}" if errors == 0 else f"no convergence in {epoch} epochs"
print(f"{status}; weights {w.round(2)}, bias {b:.2f}")
acc = np.mean([(1 if xi @ w + b > 0 else 0) == ti for xi, ti in zip(X, t)])
print(f"accuracy on 100 flowers: {acc:.2f}")
