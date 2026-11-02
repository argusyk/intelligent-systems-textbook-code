import numpy as np
from scipy.optimize import linprog

A = np.array([[5, 1], [2, 4]], float)           # payoff matrix of the security service
m, n = A.shape
res = linprog(c=np.ones(m), A_ub=-A.T, b_ub=-np.ones(n), bounds=[(0, None)] * m)
v = 1 / res.x.sum()
print("game value:", round(v, 3), "service strategy:", (res.x * v).round(3))
# game value: 3.0 service strategy: [0.333 0.667]
