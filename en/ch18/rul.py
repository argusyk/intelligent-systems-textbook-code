# Forecasting the remaining useful life (RUL) on synthetic data modelled on NASA C-MAPSS
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

rng = np.random.default_rng(0)

def simulate_unit(life):
    """One unit from the new state to failure: 4 sensors with exponential degradation."""
    t = np.arange(life)
    wear = np.exp(3 * (t / life)) - 1                        # hidden wear 0 .. e^3-1
    egt  = 600 + 4.0 * wear + rng.normal(0, 3, life)
    vib  = 2.0 + 0.25 * wear + rng.normal(0, 0.3, life)
    oil  = 4.0 - 0.06 * wear + rng.normal(0, 0.08, life)
    fuel = 120 + 1.2 * wear + rng.normal(0, 2, life)
    return np.column_stack([egt, vib, oil, fuel])

def features(S, w=10):
    """Features in a window of the last w cycles: the mean and the slope of each sensor."""
    X = []
    for k in range(len(S)):
        win = S[max(0, k - w + 1):k + 1]
        slope = np.polyfit(np.arange(len(win)), win, 1)[0] if len(win) > 1 else np.zeros(S.shape[1])
        X.append(np.concatenate([win.mean(0), slope]))
    return np.array(X)

def dataset(units, cap=125):
    X, y, groups = [], [], []
    for u, life in enumerate(units):
        S = simulate_unit(life)
        X.append(features(S)); y.append(np.minimum(np.arange(life)[::-1], cap)); groups += [u] * life
    return np.vstack(X), np.concatenate(y), np.array(groups)

def nasa_score(d):                                           # d = forecast - actual
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))

train_lives = rng.integers(150, 300, 80)
test_lives = rng.integers(150, 300, 20)
X_tr, y_tr, _ = dataset(train_lives)
X_all, y_all, g = dataset(test_lives)
cut = [rng.integers(40, life - 5) for life in test_lives]    # the unit is observed only up to the moment cut
idx = [np.where(g == u)[0][c] for u, c in enumerate(cut)]
X_te, y_te = X_all[idx], y_all[idx]

for name, m in [("linear regression", LinearRegression()),
                ("random forest", RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0))]:
    p = m.fit(X_tr, y_tr).predict(X_te)
    d = p - y_te
    print(f"{name:18s} RMSE = {np.sqrt(np.mean(d**2)):5.1f} cycles   NASA score = {nasa_score(d):7.1f}   "
          f"late forecasts {np.sum(d > 0)}/{len(d)}")
