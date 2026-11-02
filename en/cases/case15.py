# How much to invest in cybersecurity: a game against nature (annual costs, thousand UAH)
import numpy as np

acts = ["minimal protection", "basic protection", "full protection + SOC"]
states = ["no attacks", "mass attack", "targeted attack"]
C = np.array([[20, 300, 1500],
              [120, 160, 700],
              [400, 410, 480]], float)
q = np.array([0.5, 0.4, 0.1])                                # experts' probability estimates

res = {"Bayes": C @ q, "Laplace": C.mean(1), "Wald": C.max(1),
       "Savage": (C - C.min(0)).max(1), "Hurwicz (0.5)": 0.5 * C.max(1) + 0.5 * C.min(1)}
for k, v in res.items():
    print(f"{k:14s} {v.round(0)} -> {acts[int(v.argmin())]}")
for p3 in (0.1, 0.3, 0.5, 0.55):  # sensitivity to the probability of a targeted attack
    qq = np.array([0.6 - p3, 0.4, p3])
    print(f"P(targeted) = {p3:.2f}: Bayes -> {acts[int((C @ qq).argmin())]}")
