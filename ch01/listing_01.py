# Правила зберігаються як дані, а не як вкладені if у коді
RULES = [
    {"if": {"EGT_high": True, "start_slow": True}, "then": "H1: забруднені форсунки"},
    {"if": {"oil_low": True},                      "then": "H5: нестача або витік масла"},
    {"if": {"vib_high": True, "oil_low": False},   "then": "H3: знос підшипника ротора"},
]

def diagnose(facts):
    return [r["then"] for r in RULES
            if all(facts.get(k) == v for k, v in r["if"].items())]

facts = {"EGT_high": True, "start_slow": True, "oil_low": False, "vib_high": False}
print(diagnose(facts))   # ['H1: забруднені форсунки']
