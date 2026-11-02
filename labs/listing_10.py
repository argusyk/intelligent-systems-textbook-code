import numpy as np

def fairness(decision, group, y_true):
    d, g, y = map(np.asarray, (decision, group, y_true))
    sr = [d[g == k].mean() for k in (0, 1)]
    tpr = [d[(g == k) & (y == 1)].mean() for k in (0, 1)]
    return {"частки позитивних": np.round(sr, 3), "відношення": round(float(sr[1] / sr[0]),
                                                                      3),
            "рівність можливостей": np.round(tpr, 3)}

rng = np.random.default_rng(2)
g = rng.random(1000) < 0.4; y = rng.random(1000) < 0.5
d = rng.random(1000) < np.where(y, 0.8, 0.2) - 0.15 * g          # модель рідше обирає групу 1
print(fairness(d, g, y))
