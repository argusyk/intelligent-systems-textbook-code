import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

egt = ctrl.Antecedent(np.arange(400, 801, 1), "egt")
start = ctrl.Antecedent(np.arange(0, 101, 1), "start")
risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")      # centroid by default

egt["normal"] = fuzz.trapmf(egt.universe, [400, 400, 600, 640])
egt["elevated"] = fuzz.trimf(egt.universe, [600, 650, 700])
egt["high"] = fuzz.trapmf(egt.universe, [660, 720, 800, 800])
start["normal"] = fuzz.trapmf(start.universe, [0, 0, 35, 50])
start["prolonged"] = fuzz.trapmf(start.universe, [40, 60, 100, 100])
risk["low"] = fuzz.trimf(risk.universe, [0, 0, 40])
risk["medium"] = fuzz.trimf(risk.universe, [20, 50, 80])
risk["high"] = fuzz.trimf(risk.universe, [60, 100, 100])

rules = [ctrl.Rule(egt["normal"] & start["normal"], risk["low"]),
         ctrl.Rule(egt["elevated"] & start["normal"], risk["medium"]),
         ctrl.Rule(egt["high"], risk["high"]),
         ctrl.Rule(egt["elevated"] & start["prolonged"], risk["high"]),
         ctrl.Rule(start["prolonged"], risk["medium"])]

sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
sim.input["egt"], sim.input["start"] = 680, 52
sim.compute()
print(round(sim.output["risk"], 1))    # about 60.6
