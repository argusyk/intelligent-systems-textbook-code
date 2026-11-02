H = ["nozzles", "governor", "bearing", "starter", "oil", "thermocouple"]
prior = [0.30, 0.10, 0.15, 0.20, 0.15, 0.10]
LIK = {   # P(E | H_i) for each hypothesis
    "overheating":     [0.80, 0.70, 0.10, 0.05, 0.20, 0.90],
    "slow start":      [0.60, 0.30, 0.10, 0.85, 0.10, 0.05],
    "high fuel flow":  [0.70, 0.80, 0.05, 0.05, 0.05, 0.05],
}

def update(p, evidence):
    u = [pi * li for pi, li in zip(p, LIK[evidence])]
    s = sum(u)                       # P(E) by the law of total probability
    return [x / s for x in u]

p = prior
for e in ["overheating", "slow start", "high fuel flow"]:
    p = update(p, e)
    print(f"{e:16s}", " ".join(f"{x:.3f}" for x in p))
# overheating      0.527 0.154 0.033 0.022 0.066 0.198
# slow start       0.789 0.115 0.008 0.047 0.016 0.025
# high fuel flow   0.851 0.142 0.001 0.004 0.001 0.002
