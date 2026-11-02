import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

egt = ctrl.Antecedent(np.arange(400, 801, 1), "egt")
start = ctrl.Antecedent(np.arange(0, 101, 1), "start")
risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")      # центроїд за замовчуванням

egt["норма"] = fuzz.trapmf(egt.universe, [400, 400, 600, 640])
egt["підвищена"] = fuzz.trimf(egt.universe, [600, 650, 700])
egt["висока"] = fuzz.trapmf(egt.universe, [660, 720, 800, 800])
start["нормальний"] = fuzz.trapmf(start.universe, [0, 0, 35, 50])
start["затяжний"] = fuzz.trapmf(start.universe, [40, 60, 100, 100])
risk["низький"] = fuzz.trimf(risk.universe, [0, 0, 40])
risk["середній"] = fuzz.trimf(risk.universe, [20, 50, 80])
risk["високий"] = fuzz.trimf(risk.universe, [60, 100, 100])

rules = [ctrl.Rule(egt["норма"] & start["нормальний"], risk["низький"]),
         ctrl.Rule(egt["підвищена"] & start["нормальний"], risk["середній"]),
         ctrl.Rule(egt["висока"], risk["високий"]),
         ctrl.Rule(egt["підвищена"] & start["затяжний"], risk["високий"]),
         ctrl.Rule(start["затяжний"], risk["середній"])]

sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
sim.input["egt"], sim.input["start"] = 680, 52
sim.compute()
print(round(sim.output["risk"], 1))    # близько 60.6
