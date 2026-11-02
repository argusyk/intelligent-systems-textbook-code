# Conformal prediction: a guaranteed lower bound of the remaining useful life for the model from rul.py
import io, contextlib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
with contextlib.redirect_stdout(io.StringIO()):
    import rul                                                    # data and features of Section 18.4

rng = np.random.default_rng(1)
def sample(n_units, per_unit):
    """New units observed up to random moments (as in the test sample of rul.py)."""
    lives = rng.integers(150, 300, n_units)
    X_all, y_all, g = rul.dataset(lives)
    pick = [np.where(g == u)[0][rng.integers(40, life - 5)] for u, life in enumerate(lives)
            for _ in range(per_unit)]
    return X_all[pick], y_all[pick]

X_cal, y_cal = sample(30, 5)                                       # calibration sample: 150 points
X_te, y_te = sample(100, 3)                                        # coverage check: 300 points

model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0).fit(rul.X_tr, rul.y_tr)
for alpha in [0.2, 0.1, 0.05]:
    s = model.predict(X_cal) - y_cal                               # how much the forecast overestimated the life
    n = len(s)
    q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))   # conformal quantile
    lower = model.predict(X_te) - q                            # lower bound at level 1 - alpha
    cover = np.mean(y_te >= lower)
    print(f"level {1 - alpha:.0%}: margin q = {q:5.1f} cycles, test coverage {cover:.2f}, "
          f"mean lower bound {lower.mean():5.1f} with a mean forecast of {model.predict(X_te).mean():5.1f}")
