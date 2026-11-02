# Базові методи прогнозування часового ряду: наївний, ковзне середнє, експоненційне згладжування
import numpy as np

rng = np.random.default_rng(3)
t = np.arange(120)                                            # 120 днів: тренд + тижнева сезонність + шум
y = 50 + 0.2 * t + 6 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, t.size)
train, test = y[:100], y[100:]

def ses(series, alpha):                                       # просте експоненційне згладжування
    level = series[0]
    for v in series[1:]:
        level = alpha * v + (1 - alpha) * level
    return level

def holt(series, alpha, beta):                                # згладжування з трендом (Холт)
    level, trend = series[0], series[1] - series[0]
    for v in series[1:]:
        prev = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return level, trend

h = np.arange(1, 21)
fc = {"наївний": np.full(20, train[-1]),
      "ковзне середнє (7)": np.full(20, train[-7:].mean()),
      "сезонний наївний": np.array([train[-7 + (k - 1) % 7] for k in h]),
      "експ. згладжування": np.full(20, ses(train, 0.3))}
lv, tr = holt(train, 0.3, 0.1)
fc["Холт (тренд)"] = lv + tr * h
for name, f in fc.items():
    print(f"{name:20s} MAE = {np.mean(np.abs(f - test)):.2f}")
