# Пояснення прогнозу прогресування діабету на реальних клінічних даних (набір Diabetes, 442
# пацієнти)
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.model_selection import train_test_split

d = load_diabetes()
names = ["вік", "стать", "ІМТ", "тиск", "s1", "s2", "s3", "s4", "s5 (тригліцериди)",
         "s6 (глюкоза)"]
X_tr, X_te, y_tr, y_te = train_test_split(d.data, d.target, test_size=0.3, random_state=0)
gb = GradientBoostingRegressor(max_depth=2, n_estimators=150, learning_rate=0.05,
                               random_state=0).fit(X_tr, y_tr)
print(f"R^2 на тесті: {gb.score(X_te, y_te):.3f}")
imp = permutation_importance(gb, X_te, y_te, n_repeats=30, random_state=0)
top = np.argsort(-imp.importances_mean)[:4]
print("найважливіші ознаки:", [(names[i], round(float(imp.importances_mean[i]),
                                                3)) for i in top])
pd = partial_dependence(gb, X_tr, [2], grid_resolution=5, method="brute")
print("часткова залежність від ІМТ:", np.round(pd["average"][0], 0))

x = X_te[np.argmax(gb.predict(X_te))]  # пацієнт з найгіршим прогнозом
sd = d.data.std(0)
best = None
for a in np.linspace(0, 2, 21):  # контрфакт: зниження ІМТ і s5
    for b in np.linspace(0, 2, 21):  # (у стандартних відхиленнях)
        x2 = x.copy(); x2[2] -= a * sd[2]; x2[8] -= b * sd[8]
        if gb.predict(x2[None])[0] < 200 and (best is None or a + b < best[0] + best[1]):
            best = (a, b)
print(f"прогноз {gb.predict(x[None])[0]:.0f}; нижче 200, якщо знизити ІМТ на {best[0]:.1f} і "
      f"тригліцериди на {best[1]:.1f} стандартного відхилення")
