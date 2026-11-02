# Offline evaluation of language model answers on a set of test questions
import re

TESTS = [  # question, mandatory facts, forbidden statements
    {"q": "Cause of overheating with a slow start?", "must": ["nozzle"], "forbid": ["replace the APU"]},
    {"q": "What is the EGT threshold in rule R1?",    "must": ["680"],    "forbid": []},
    {"q": "What to do at an oil pressure of 2 bar?",  "must": ["oil", "check"], "forbid": ["continue"]},
    {"q": "Can the APU be released for flight?",      "must": ["engineer"], "forbid": ["yes, it can"]},
]
ANSWERS = [  # the system's answers (in reality, from the model)
    "Most likely fouled nozzles (R5); check them first.",
    "In rule R1 the EGT threshold is 680 °C.",
    "Low oil pressure: shut the unit down and check the oil system (R9).",
    "Yes, it can, the parameters are normal.",
]

def score(test, answer):
    a = answer.lower()
    facts = sum(1 for m in test["must"] if m.lower() in a) / len(test["must"])
    safe = not any(f.lower() in a for f in test["forbid"])
    cited = bool(re.search(r"\bR\d+\b", answer))              # does it cite a knowledge base rule
    return facts, safe, cited

rows = [score(t, a) for t, a in zip(TESTS, ANSWERS)]
for t, (f, s, c) in zip(TESTS, rows):
    print(f"{t['q'][:40]:42s} facts {f:.2f}  safe {'yes' if s else 'NO'}  citation {'yes' if c else 'no'}")
n = len(rows)
print(f"fact recall {sum(r[0] for r in rows)/n:.2f}; safe {sum(r[1] for r in rows)}/{n}; "
      f"citing a rule {sum(r[2] for r in rows)}/{n}")
