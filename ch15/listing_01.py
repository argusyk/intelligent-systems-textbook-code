import numpy as np
from scipy.optimize import linprog

A = np.array([[5, 1], [2, 4]], float)           # платіжна матриця служби
m, n = A.shape
res = linprog(c=np.ones(m), A_ub=-A.T, b_ub=-np.ones(n), bounds=[(0, None)] * m)
v = 1 / res.x.sum()
print("ціна гри:", round(v, 3), "стратегія служби:", (res.x * v).round(3))
# ціна гри: 3.0 стратегія служби: [0.333 0.667]
