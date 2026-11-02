import numpy as np

def criteria(C, q=None, alpha=0.5):
    C = np.asarray(C, float)
    res = {"Лапласа": C.mean(1), "Вальда": C.max(1),
           "Севіджа": (C - C.min(0)).max(1),
           f"Гурвіца ({alpha})": alpha * C.max(1) + (1 - alpha) * C.min(1)}
    if q is not None:
        res["Байєса"] = C @ np.asarray(q, float)
    return {k: (v.round(2), int(v.argmin()) + 1) for k, v in res.items()}

C = [[12, 40, 95], [30, 35, 50], [60, 62, 65]]      # матриця рис. 15.6
for k, (vals, best) in criteria(C, q=[0.6, 0.3, 0.1]).items():
    print(f"{k:14s} {vals}  ->  A{best}")
