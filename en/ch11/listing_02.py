# Choosing a cloud platform for a student project with the analytic hierarchy process
import numpy as np

def ahp(M):
    M = np.array(M, float); n = len(M)
    w = M.prod(axis=1) ** (1 / n); w /= w.sum()
    lam = float(np.mean(M @ w / w))
    return w, ((lam - n) / (n - 1)) / {3: 0.58, 4: 0.90}[n]

crit = ["cost", "free tier", "ease of use", "AI services"]
C = [[1, 1/2, 2, 3], [2, 1, 3, 4], [1/2, 1/3, 1, 2], [1/3, 1/4, 1/2, 1]]
alts = ["Platform A", "Platform B", "Own server"]
A = {"cost": [[1, 1, 1/3], [1, 1, 1/3], [3, 3, 1]],
     "free tier": [[1, 3, 5], [1/3, 1, 3], [1/5, 1/3, 1]],
     "ease of use": [[1, 2, 5], [1/2, 1, 4], [1/5, 1/4, 1]],
     "AI services": [[1, 1/2, 7], [2, 1, 8], [1/7, 1/8, 1]]}

wc, cr = ahp(C)
print("criteria weights:", dict(zip(crit, wc.round(3).tolist())), "CR =", round(cr, 3))
W = np.column_stack([ahp(A[c])[0] for c in crit])
for name, s in sorted(zip(alts, W @ wc), key=lambda t: -t[1]):
    print(f"{name:16s} {s:.3f}")
print("worst CR among the alternative matrices:", round(max(ahp(A[c])[1] for c in crit), 3))
