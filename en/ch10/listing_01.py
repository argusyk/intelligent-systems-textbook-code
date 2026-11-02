def combine(a, b):
    if a >= 0 and b >= 0:
        return a + b * (1 - a)
    if a < 0 and b < 0:
        return a + b * (1 + a)
    return (a + b) / (1 - min(abs(a), abs(b)))

def rule_cf(conds, rule, facts, threshold=0.2):
    cf_e = min(facts.get(c, 0.0) for c in conds)       # conjunction of conditions
    return cf_e * rule if cf_e > threshold else None    # the rule did not fire

facts = {"C1": 0.9, "C2": 0.6, "fuel_high": 0.8, "washed_recently": 1.0}
rules_H1 = [(["C1", "C2"], 0.7), (["fuel_high"], 0.5), (["washed_recently"], -0.6)]

cf = 0.0
for conds, r in rules_H1:
    c = rule_cf(conds, r, facts)
    if c is not None:
        cf = combine(cf, c)
        print(f"{conds} -> {c:+.2f}; total {cf:+.3f}")
# ['C1', 'C2'] -> +0.42; total +0.420
# ['fuel_high'] -> +0.40; total +0.652
# ['washed_recently'] -> -0.60; total +0.130
