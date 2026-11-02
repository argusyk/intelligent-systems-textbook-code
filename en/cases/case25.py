# Auditing a CV pre-screening system: does it treat two groups of candidates
# in the same way?
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(11)
n = 6000
group = rng.random(n) < 0.35                                      # group B (35 % of candidates)
skill = rng.normal(0, 1, n)                                       # true qualification
# career break (more frequent in group B)
gap = (rng.random(n) < np.where(group, 0.45, 0.10)).astype(float)
exp_years = np.clip(3 + 2 * skill + rng.normal(0, 1.5, n) - 1.5 * gap, 0, None)
# historical human decisions
hired_hist = (skill + 0.5 * rng.normal(0, 1, n) - 0.8 * gap > 0.3).astype(int)
success = (skill + 0.5 * rng.normal(0, 1, n) > 0.3).astype(int)  # actual job performance
X = np.column_stack([exp_years, gap, skill + rng.normal(0, 0.7, n)])  # CV features + test
tr = np.arange(n) < 4000
m = LogisticRegression().fit(X[tr], hired_hist[tr])  # training on historical decisions
p = m.predict_proba(X[~tr])[:, 1]; g = group[~tr]; y = success[~tr]

def report(sel, name):
    sr = [sel[g == k].mean() for k in (False, True)]
    tpr = [sel[(g == k) & (y == 1)].mean() for k in (False, True)]
    print(f"{name:26s} selected A/B {sr[0]:.2f}/{sr[1]:.2f} (ratio {sr[1] / sr[0]:.2f}); "
          f"successful selected A/B {tpr[0]:.2f}/{tpr[1]:.2f}")
report(p >= 0.5, "model, common threshold")
m2 = LogisticRegression().fit(X[tr][:, [0, 2]], hired_hist[tr])  # without the "break" feature
report(m2.predict_proba(X[~tr][:, [0, 2]])[:, 1] >= 0.5, "without the 'break' feature")
m3 = LogisticRegression().fit(X[tr][:, [2]], success[tr])  # target = performance, test only
report(m3.predict_proba(X[~tr][:, [2]])[:, 1] >= 0.5, "target = performance, test")
