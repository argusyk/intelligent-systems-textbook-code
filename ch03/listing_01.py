KB = [  # база знань: (ім'я, умови, висновок)
    ("R1", {"EGT_high", "start_slow"},  "fuel_nozzles_dirty"),
    ("R2", {"oil_pressure_low"},        "oil_leak_suspected"),
    ("R3", {"fuel_nozzles_dirty"},      "action_clean_nozzles"),
    ("R4", {"oil_leak_suspected"},      "action_check_oil_lines"),
]

def forward_chain(facts):
    wm = set(facts)          # робоча пам'ять
    trace = []               # протокол для підсистеми пояснень
    while True:
        conflict = [r for r in KB
                    if r[1] <= wm and r[2] not in wm]   # рефракція
        if not conflict:
            break
        name, cond, concl = conflict[0]                  # порядок запису
        wm.add(concl)
        trace.append(f"{name}: {sorted(cond)} -> {concl}")
    return wm, trace

wm, trace = forward_chain({"EGT_high", "start_slow"})
print(*trace, sep="\n")
# R1: ['EGT_high', 'start_slow'] -> fuel_nozzles_dirty
# R3: ['fuel_nozzles_dirty'] -> action_clean_nozzles
