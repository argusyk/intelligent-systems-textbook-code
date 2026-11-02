import numpy as np

def ahp(M):
    M = np.array(M, float)
    n = len(M)
    g = M.prod(axis=1) ** (1 / n)           # середнє геометричне рядків
    w = g / g.sum()                         # формула (11.4)
    lam = float(np.mean(M @ w / w))         # оцінка lambda_max
    ci = (lam - n) / (n - 1)
    return w, ci / {3: 0.58, 4: 0.90, 5: 1.12}[n]   # ваги і CR

crit = [[1, 1/2, 1/5], [2, 1, 1/3], [5, 3, 1]]
alts = {"вартість": [[1, 3, 7], [1/3, 1, 4], [1/7, 1/4, 1]],
        "простій":  [[1, 1/2, 3], [2, 1, 5], [1/3, 1/5, 1]],
        "надійність": [[1, 1/3, 1/5], [3, 1, 1/2], [5, 2, 1]]}

wc, cr = ahp(crit)
W = np.column_stack([ahp(m)[0] for m in alts.values()])
print("ваги критеріїв:", wc.round(3), "CR =", round(cr, 4))
print("пріоритети A1..A3:", (W @ wc).round(3))   # [0.222 0.366 0.412]
