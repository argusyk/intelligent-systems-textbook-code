# Розклад робіт нічної зміни як задача задоволення обмежень: пошук з поверненням,
# попереджувальна перевірка (forward checking) та евристика «найменше значень» (MRV)
TASKS = {  # робота: (бригада, тривалість у годинах)
    "панелі": ("механіки", 1), "форсунки": ("механіки", 2), "стартер": ("механіки", 1),
    "фільтр": ("механіки", 1), "проводка": ("електрики", 2), "датчики": ("електрики", 1),
    "термопара": ("електрики", 1), "пробний запуск": ("механіки", 1)}
SLOTS = 8                                          # зміна з 22:00 до 06:00
BEFORE = [("панелі", "форсунки"), ("панелі", "стартер"), ("форсунки", "пробний запуск"),
          ("стартер", "пробний запуск"), ("фільтр", "пробний запуск"),
          ("проводка", "пробний запуск"), ("датчики", "пробний запуск"),
          ("термопара", "пробний запуск")]
STAND = {"форсунки", "датчики", "термопара"}       # один випробувальний стенд

def end(t, s): return s + TASKS[t][1]
def ok(a, sa, b, sb):
    """Усі обмеження між двома роботами з призначеними годинами початку."""
    overlap = sa < end(b, sb) and sb < end(a, sa)
    if overlap and TASKS[a][0] == TASKS[b][0]: return False         # одна бригада
    if overlap and {a, b} <= STAND: return False                     # один стенд
    if (a, b) in BEFORE and end(a, sa) > sb: return False            # передування
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
for name, mrv, fc in [("пошук з поверненням", False, False),
                      ("+ попереджувальна перевірка", False, True),
                      ("+ перевірка і MRV", True, True)]:
    stats, found = [0], []
    solve({}, dom0, stats, mrv, fc, found)
    print(f"{name:28s} розкладів: {len(found):4d}   перевірено призначень: {stats[0]:6d}")
best = min(found, key=lambda a: max(end(t, s) for t, s in a.items()))
print("найкоротший розклад, годин:", max(end(t, s) for t, s in best.items()))
for t, s in sorted(best.items(), key=lambda x: (x[1], x[0])):
    print(f"  {(22 + s) % 24:02d}:00-{(22 + end(t, s)) % 24:02d}:00  {t:15s} ({TASKS[t][0]})")
