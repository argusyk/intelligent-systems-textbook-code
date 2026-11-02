level, target, valve = 30.0, 50.0, 0.0
history = []
for k in range(50):
    e = level - target
    rate = level - history[-1] if history else 0.0   # швидкість зміни рівня
    sim.input["error"], sim.input["rate"] = e, rate   # sim: ControlSystemSimulation
    sim.compute()
    valve = max(0.0, min(100.0, valve + 0.1 * sim.output["valve"]))
    history.append(level)
    level += 0.05 * valve - 1.5                      # притік мінус сталий відтік
print(round(level, 1))
