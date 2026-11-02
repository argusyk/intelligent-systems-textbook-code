import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
t = np.array([[0], [1], [1], [0]], float)
rng = np.random.default_rng(1)
W1, b1 = rng.normal(0, 1, (2, 4)), np.zeros(4)      # 2 входи -> 4 приховані
W2, b2 = rng.normal(0, 1, (4, 1)), np.zeros(1)      # 4 приховані -> 1 вихід
sig = lambda z: 1 / (1 + np.exp(-z))

for epoch in range(5000):
    h = np.tanh(X @ W1 + b1)                        # прямий прохід
    y = sig(h @ W2 + b2)
    d2 = y - t  # помилка виходу (сигмоїда + крос-ентропія)
    d1 = (d2 @ W2.T) * (1 - h ** 2)                 # формула (14.6), похідна tanh
    W2 -= 0.5 * h.T @ d2 / 4;  b2 -= 0.5 * d2.mean(0)     # формула (14.7)
    W1 -= 0.5 * X.T @ d1 / 4;  b1 -= 0.5 * d1.mean(0)

print(y.round(3).ravel())    # близько [0, 1, 1, 0]
