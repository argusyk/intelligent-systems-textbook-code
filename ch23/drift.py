# Моніторинг дрейфу даних: індекс стабільності популяції (PSI) і тест Колмогорова – Смирнова
import numpy as np
from scipy.stats import ks_2samp
from train_and_package import make_data, FEATURES

ref = np.load("train_reference.npy")

def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))

for month, shift in [("січень", 0), ("квітень", 8), ("липень (спека)", 25)]:
    new, _ = make_data(300, seed=100 + shift, egt_shift=shift)          # у спеку EGT вища на shift °C
    row = []
    for i, f in enumerate(FEATURES[:2]):
        p = psi(ref[:, i], new[:, i]); ks = ks_2samp(ref[:, i], new[:, i]).pvalue
        flag = "ДРЕЙФ" if p > 0.2 else ("увага" if p > 0.1 else "норма")
        row.append(f"{f}: PSI={p:.3f} p={ks:.3f} {flag}")
    print(f"{month:15s} " + " | ".join(row))
