RULES = {   # conclusion -> list of alternative sets of conditions (OR of AND)
    "H1_nozzles": [["C1_overheat", "C2_slow_start"]],
    "C1_overheat": [["egt_gt_680"]],
    "C2_slow_start": [["start_gt_45"]],
}
ASKABLE = {"egt_gt_680": "Does EGT exceed 680 °C?", "start_gt_45": "Is the start longer than 45 s?"}

def prove(goal, known, stack):
    if goal in known:
        return known[goal]
    for conds in RULES.get(goal, []):                  # OR node
        if all(prove(c, known, stack + [goal]) for c in conds):   # AND node
            known[goal] = True
            return True
    if goal in ASKABLE:
        while True:
            ans = input(ASKABLE[goal] + " (yes/no/why): ").strip()
            if ans == "why":
                print("  Needed for:", " <- ".join(reversed(stack)))
                continue
            known[goal] = (ans == "yes")
            return known[goal]
    known[goal] = False
    return False

print("H1 confirmed" if prove("H1_nozzles", {}, []) else "H1 not confirmed")
