# Explaining model decisions: permutation importance, partial dependence, Shapley values, local surrogate
import itertools, math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

rng = np.random.default_rng(42)
F = ["egt", "start_time", "fuel_flow", "vibration"]
n = 800
fault = rng.random(n) < 0.3
X = np.column_stack([rng.normal(610, 25, n) + fault * rng.normal(55, 20, n),
                     rng.normal(36, 6, n) + fault * rng.normal(10, 6, n),
                     rng.normal(118, 8, n) + fault * rng.normal(18, 8, n),
                     rng.normal(3.0, 0.8, n)])
y = fault.astype(int)
model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0).fit(X[:600], y[:600])
f = lambda Z: model.predict_proba(Z)[:, 1]

# 1) globally: permutation importance on the test part
imp = permutation_importance(model, X[600:], y[600:], scoring="roc_auc", n_repeats=20, random_state=0)
print("permutation importance (ROC AUC drop):", dict(zip(F, imp.importances_mean.round(3).tolist())))

# 2) partial dependence on EGT: the mean prediction at a fixed EGT
grid = np.linspace(560, 740, 7)
pd_vals = [f(np.column_stack([np.full(200, g), X[:200, 1:]])).mean() for g in grid]
print("partial dependence of P(fault) on EGT:", dict(zip(grid.astype(int).tolist(), np.round(pd_vals, 2).tolist())))

# 3) exact Shapley values for one prediction (4 features -> 16 coalitions)
x0 = np.array([700, 40, 121, 3.2])                   # high EGT, other parameters almost normal
background = X[:200]
def v(S):                                            # coalition value: the mean prediction when the features S are taken from x0
    Z = background.copy(); Z[:, list(S)] = x0[list(S)]
    return f(Z).mean()
phi = np.zeros(4)
for i in range(4):
    others = [j for j in range(4) if j != i]
    for r in range(4):
        for S in itertools.combinations(others, r):
            w = math.factorial(r) * math.factorial(4 - r - 1) / math.factorial(4)
            phi[i] += w * (v(S + (i,)) - v(S))
base = v(()); pred = f(x0[None])[0]
print(f"base value {base:.3f}; prediction {pred:.3f}; Shapley sum {base + phi.sum():.3f}")
print("Shapley values:", dict(zip(F, phi.round(3).tolist())))

# 4) local surrogate (the LIME idea): a linear model on perturbations near x0, weighted by proximity
Zs = x0 + rng.normal(0, 1, (500, 4)) * X.std(0) * 0.5
w = np.exp(-np.sum(((Zs - x0) / X.std(0)) ** 2, axis=1) / 2)
lime = Ridge(alpha=1.0).fit((Zs - x0) / X.std(0), f(Zs), sample_weight=w)
print("local surrogate, coefficients per 1 standard deviation:", dict(zip(F, lime.coef_.round(3).tolist())))

# 5) global surrogate: a depth-3 tree trained to repeat the forest
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[:600], (f(X[:600]) > 0.5).astype(int))
fid = np.mean(tree.predict(X[600:]) == (f(X[600:]) > 0.5))
print(f"accuracy of reproducing the forest by the surrogate tree on the test: {fid:.3f}")
