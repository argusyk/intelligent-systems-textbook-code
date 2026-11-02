RULES = {   # висновок -> список альтернативних наборів умов (АБО з І)
    "H1_nozzles": [["C1_overheat", "C2_slow_start"]],
    "C1_overheat": [["egt_gt_680"]],
    "C2_slow_start": [["start_gt_45"]],
}
ASKABLE = {"egt_gt_680": "EGT перевищує 680 °C?", "start_gt_45": "Запуск довший за 45 с?"}

def prove(goal, known, stack):
    if goal in known:
        return known[goal]
    for conds in RULES.get(goal, []):                  # АБО-вузол
        if all(prove(c, known, stack + [goal]) for c in conds):   # І-вузол
            known[goal] = True
            return True
    if goal in ASKABLE:
        while True:
            ans = input(ASKABLE[goal] + " (так/ні/чому): ").strip()
            if ans == "чому":
                print("  Потрібно для:", " <- ".join(reversed(stack)))
                continue
            known[goal] = (ans == "так")
            return known[goal]
    known[goal] = False
    return False

print("H1 підтверджено" if prove("H1_nozzles", {}, []) else "H1 не підтверджено")
