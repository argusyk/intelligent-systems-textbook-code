# Night-shift work schedule as a constraint satisfaction problem: backtracking search,
# forward checking and the "minimum remaining values" (MRV) heuristic
TASKS = {  # job: (crew, duration in hours)
    "panels": ("mechanics", 1), "nozzles": ("mechanics", 2), "starter": ("mechanics", 1),
    "filter": ("mechanics", 1), "wiring": ("electricians", 2), "sensors": ("electricians", 1),
    "thermocouple": ("electricians", 1), "test run": ("mechanics", 1)}
SLOTS = 8                                          # shift from 22:00 to 06:00
BEFORE = [("panels", "nozzles"), ("panels", "starter"), ("nozzles", "test run"),
          ("starter", "test run"), ("filter", "test run"),
          ("wiring", "test run"), ("sensors", "test run"),
          ("thermocouple", "test run")]
STAND = {"nozzles", "sensors", "thermocouple"}       # a single test bench

def end(t, s): return s + TASKS[t][1]
def ok(a, sa, b, sb):
    """All constraints between two jobs with assigned start hours."""
    overlap = sa < end(b, sb) and sb < end(a, sa)
    if overlap and TASKS[a][0] == TASKS[b][0]: return False         # same crew
    if overlap and {a, b} <= STAND: return False                     # same bench
    if (a, b) in BEFORE and end(a, sa) > sb: return False            # precedence
    if (b, a) in BEFORE and end(b, sb) > sa: return False
    return True

def solve(assign, dom, stats, mrv, fc, found):
    if len(assign) == len(TASKS):
        found.append(dict(assign)); return
    free = [t for t in TASKS if t not in assign]
    var = min(free, key=lambda t: len(dom[t])) if mrv else free[0]
    for s in dom[var]:
        stats[0] += 1
        if all(ok(var, s, t, st) for t, st in assign.items()):
            assign[var] = s
            new = {t: [v for v in dom[t] if ok(var, s, t,
                                               v)] for t in free if t != var} if fc else dom
            if all(new[t] for t in free if t != var):
                solve(assign, {**dom, **new} if fc else dom, stats, mrv, fc, found)
            del assign[var]

dom0 = {t: [s for s in range(SLOTS) if end(t, s) <= SLOTS] for t in TASKS}
for name, mrv, fc in [("backtracking search", False, False),
                      ("+ forward checking", False, True),
                      ("+ forward checking and MRV", True, True)]:
    stats, found = [0], []
    solve({}, dom0, stats, mrv, fc, found)
    print(f"{name:28s} schedules: {len(found):4d}   assignments checked: {stats[0]:6d}")
best = min(found, key=lambda a: max(end(t, s) for t, s in a.items()))
print("shortest schedule, hours:", max(end(t, s) for t, s in best.items()))
for t, s in sorted(best.items(), key=lambda x: (x[1], x[0])):
    print(f"  {(22 + s) % 24:02d}:00-{(22 + end(t, s)) % 24:02d}:00  {t:15s} ({TASKS[t][0]})")
