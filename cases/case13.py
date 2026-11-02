# Виявлення аномалій у телеметрії вебсервера без учителя (Isolation Forest)
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(7)
hours = np.arange(24 * 14)                                   # два тижні погодинно
load = 40 + 25 * np.sin(2 * np.pi * (hours % 24 - 8) / 24).clip(0) + rng.normal(0, 4, hours.size)
latency = 80 + 1.2 * load + rng.normal(0, 8, hours.size)
incidents = [100, 101, 250]                                  # штучні збої
latency[incidents] += [220, 260, 180]
load[250] = 5                                                # сервер «завис»: навантаження впало

X = np.column_stack([load, latency, hours % 24])
iso = IsolationForest(contamination=0.01, random_state=0).fit(X)
found = np.where(iso.predict(X) == -1)[0]
print("знайдені аномальні години:", found.tolist())
print("виявлено штучних збоїв:", sorted(set(found) & set(incidents)), "з", incidents)
