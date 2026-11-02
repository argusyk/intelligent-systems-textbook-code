KB = [  # knowledge base: (name, conditions, conclusion)
    ("R1", {"EGT_high", "start_slow"},  "fuel_nozzles_dirty"),
    ("R2", {"oil_pressure_low"},        "oil_leak_suspected"),
    ("R3", {"fuel_nozzles_dirty"},      "action_clean_nozzles"),
    ("R4", {"oil_leak_suspected"},      "action_check_oil_lines"),
]

def forward_chain(facts):
    wm = set(facts)          # working memory
    trace = []               # log for the explanation subsystem
    while True:
        conflict = [r for r in KB
                    if r[1] <= wm and r[2] not in wm]   # refraction
        if not conflict:
            break
        name, cond, concl = conflict[0]                  # order of writing
        wm.add(concl)
        trace.append(f"{name}: {sorted(cond)} -> {concl}")
    return wm, trace

wm, trace = forward_chain({"EGT_high", "start_slow"})
print(*trace, sep="\n")
# R1: ['EGT_high', 'start_slow'] -> fuel_nozzles_dirty
# R3: ['fuel_nozzles_dirty'] -> action_clean_nozzles
