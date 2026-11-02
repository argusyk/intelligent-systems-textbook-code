import numpy as np

def split_units(units, rng, frac=(0.6, 0.2)):
    """Splitting unit identifiers into training, calibration and test sets."""
    u = rng.permutation(np.unique(units))
    a, b = int(frac[0] * len(u)), int(sum(frac) * len(u))
    return set(u[:a]), set(u[a:b]), set(u[b:])

def conformal_lower(pred_cal, y_cal, pred_new, alpha=0.1):
    s = np.asarray(pred_cal) - np.asarray(y_cal)                 # overestimation of the RUL by the model
    n = len(s); q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    return np.asarray(pred_new) - q

rng = np.random.default_rng(0)
tr, cal, te = split_units(np.repeat(np.arange(100), 50), rng)
print(len(tr), len(cal), len(te), "units; intersection:", tr & te)
y_cal = rng.uniform(0, 125, 200); pred_cal = y_cal + rng.normal(0, 15, 200)
print("lower bounds for forecasts of 80 and 40:", conformal_lower(pred_cal, y_cal, [80,
                                                                                    40]).round(1))
