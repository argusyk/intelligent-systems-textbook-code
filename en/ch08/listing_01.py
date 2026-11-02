import numpy as np

V = ["EGT>680", "ts>45", "C1", "C2", "H1", "H2"]
E = [(0, 2), (1, 3), (2, 4), (3, 4), (2, 5)]          # arcs e1..e5

n, m = len(V), len(E)
A = np.zeros((n, n), dtype=int)                        # adjacency matrix
B = np.zeros((n, m), dtype=int)                        # incidence matrix
for j, (u, v) in enumerate(E):
    A[u, v] = 1
    B[u, j], B[v, j] = 1, -1

print(A)
print(B.sum(axis=1))    # [ 1  1  1  0 -2 -1]: outgoing minus incoming arcs
