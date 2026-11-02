import numpy as np

def psi(ref, new, bins=10):
    edges = np.quantile(ref, np.linspace(0, 1,
                                         bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    a = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    b = np.clip(np.histogram(new, edges)[0] / len(new), 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))

rng = np.random.default_rng(1)
ref = rng.beta(2, 5, 5000)
for name, new in [("той самий розподіл", rng.beta(2, 5, 2000)), ("помірний зсув",
                                                                 rng.beta(2.5, 5, 2000)),
                  ("сильний зсув", rng.beta(3.5, 4, 2000))]:
    print(f"{name:20s} PSI = {psi(ref, new):.3f}")
