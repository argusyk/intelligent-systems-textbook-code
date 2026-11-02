# Basic time series forecasting methods: naive, moving average, exponential smoothing
import numpy as np

rng = np.random.default_rng(3)
t = np.arange(120)                                            # 120 days: trend + weekly seasonality + noise
y = 50 + 0.2 * t + 6 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, t.size)
train, test = y[:100], y[100:]

def ses(series, alpha):                                       # simple exponential smoothing
    level = series[0]
    for v in series[1:]:
        level = alpha * v + (1 - alpha) * level
    return level

def holt(series, alpha, beta):                                # smoothing with a trend (Holt)
    level, trend = series[0], series[1] - series[0]
    for v in series[1:]:
        prev = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return level, trend

h = np.arange(1, 21)
fc = {"naive": np.full(20, train[-1]),
      "moving average (7)": np.full(20, train[-7:].mean()),
      "seasonal naive": np.array([train[-7 + (k - 1) % 7] for k in h]),
      "exp. smoothing": np.full(20, ses(train, 0.3))}
lv, tr = holt(train, 0.3, 0.1)
fc["Holt (trend)"] = lv + tr * h
for name, f in fc.items():
    print(f"{name:20s} MAE = {np.mean(np.abs(f - test)):.2f}")
