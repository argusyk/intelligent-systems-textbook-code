# Fuzzy risk assessment of a consumer loan (Mamdani, scikit-fuzzy)
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

debt = ctrl.Antecedent(np.arange(0, 101, 1), "debt")        # share of payments in income, %
hist = ctrl.Antecedent(np.arange(0, 11, 0.1), "history")    # credit history, score 0..10
risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

debt["low"] = fuzz.trapmf(debt.universe, [0, 0, 20, 35])
debt["medium"] = fuzz.trimf(debt.universe, [25, 40, 55])
debt["high"] = fuzz.trapmf(debt.universe, [45, 60, 100, 100])
hist["bad"] = fuzz.trapmf(hist.universe, [0, 0, 3, 5])
hist["good"] = fuzz.trapmf(hist.universe, [4, 7, 10, 10])
risk["low"] = fuzz.trimf(risk.universe, [0, 0, 40])
risk["medium"] = fuzz.trimf(risk.universe, [25, 50, 75])
risk["high"] = fuzz.trimf(risk.universe, [60, 100, 100])

rules = [ctrl.Rule(debt["low"] & hist["good"], risk["low"]),
         ctrl.Rule(debt["medium"] & hist["good"], risk["medium"]),
         ctrl.Rule(debt["low"] & hist["bad"], risk["medium"]),
         ctrl.Rule(debt["high"] | (debt["medium"] & hist["bad"]), risk["high"])]
sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

for d, h in [(15, 9), (38, 8), (38, 4), (70, 9)]:
    sim.input["debt"], sim.input["history"] = d, h
    sim.compute()
    print(f"debt {d:3d} %, history {h:2d}: risk {sim.output['risk']:.1f}")
