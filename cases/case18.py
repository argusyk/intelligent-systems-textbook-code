# Прогноз погодинного споживання електроенергії гуртожитку на добу вперед
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(4)
H = 24 * 7 * 12                                               # 12 тижнів погодинно
h = np.arange(H); hour, dow = h % 24, (h // 24) % 7
# похолодання
temp = 5 + 8 * np.sin(2 * np.pi * (h / 24 - 9) / 24) - h / 400 + rng.normal(0, 1.5, H)
load = (120 + 60 * np.exp(-((hour - 20) / 3) ** 2) + 25 * np.exp(-((hour - 8) / 2) ** 2)
        - 30 * (dow >= 5) * (hour < 11) + 2.5 * np.maximum(12 - temp, 0) + rng.normal(0, 8,
                                                                                      H))

def lagged(y, t):  # ознаки для години t: лише минуле
    return [y[t - 24], y[t - 168], y[t - 24:t].mean(), hour[t], dow[t], temp[t]]
start, split = 168, H - 24 * 14                               # останні 2 тижні – перевірка
X = np.array([lagged(load, t) for t in range(start, H)]); y = load[start:]
tr = np.arange(start, H) < split
gb = GradientBoostingRegressor(random_state=0).fit(X[tr], y[tr])
mae = lambda p: np.mean(np.abs(p - y[~tr]))
print(f"MAE, кВт·год: наївний «як учора» {mae(X[~tr, 0]):.1f}; «як тиждень тому» "
      f"{mae(X[~tr, 1]):.1f}; "
      f"бустинг {mae(gb.predict(X[~tr])):.1f}")
q_in = np.quantile(np.abs(y[tr] - gb.predict(X[tr])), 0.9)      # залишки на навчальних даних
t_idx = np.arange(start, H)
# останній тиждень – калібрування
fit_part, cal = t_idx < split - 168, (t_idx >= split - 168) & tr
gb2 = GradientBoostingRegressor(random_state=0).fit(X[fit_part], y[fit_part])
q_cal = np.quantile(np.abs(y[cal] - gb2.predict(X[cal])), 0.9)
for name, m, q in [("залишки навчання", gb, q_in), ("відкладений тиждень", gb2, q_cal)]:
    cover = np.mean(np.abs(y[~tr] - m.predict(X[~tr])) <= q)
    print(f"{name:20s} інтервал ±{q:4.1f} кВт·год, покриття на перевірці {cover:.2f} (ціль "
          f"0,90)")
