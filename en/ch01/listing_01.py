# Rules are stored as data, not as nested ifs in the code
RULES = [
    {"if": {"EGT_high": True, "start_slow": True}, "then": "H1: dirty fuel nozzles"},
    {"if": {"oil_low": True},                      "then": "H5: low oil level or oil leak"},
    {"if": {"vib_high": True, "oil_low": False},   "then": "H3: rotor bearing wear"},
]

def diagnose(facts):
    return [r["then"] for r in RULES
            if all(facts.get(k) == v for k, v in r["if"].items())]

facts = {"EGT_high": True, "start_slow": True, "oil_low": False, "vib_high": False}
print(diagnose(facts))   # ['H1: dirty fuel nozzles']
