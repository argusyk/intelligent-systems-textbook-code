# Нечітка оцінка ризику споживчого кредиту (Мамдані, scikit-fuzzy)
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

debt = ctrl.Antecedent(np.arange(0, 101, 1), "debt")        # частка платежів у доході, %
hist = ctrl.Antecedent(np.arange(0, 11, 0.1), "history")    # кредитна історія, бали 0..10
risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

debt["низька"] = fuzz.trapmf(debt.universe, [0, 0, 20, 35])
debt["середня"] = fuzz.trimf(debt.universe, [25, 40, 55])
debt["висока"] = fuzz.trapmf(debt.universe, [45, 60, 100, 100])
hist["погана"] = fuzz.trapmf(hist.universe, [0, 0, 3, 5])
hist["добра"] = fuzz.trapmf(hist.universe, [4, 7, 10, 10])
risk["низький"] = fuzz.trimf(risk.universe, [0, 0, 40])
risk["середній"] = fuzz.trimf(risk.universe, [25, 50, 75])
risk["високий"] = fuzz.trimf(risk.universe, [60, 100, 100])

rules = [ctrl.Rule(debt["низька"] & hist["добра"], risk["низький"]),
         ctrl.Rule(debt["середня"] & hist["добра"], risk["середній"]),
         ctrl.Rule(debt["низька"] & hist["погана"], risk["середній"]),
         ctrl.Rule(debt["висока"] | (debt["середня"] & hist["погана"]), risk["високий"])]
sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

for d, h in [(15, 9), (38, 8), (38, 4), (70, 9)]:
    sim.input["debt"], sim.input["history"] = d, h
    sim.compute()
    print(f"борг {d:3d} %, історія {h:2d}: ризик {sim.output['risk']:.1f}")
