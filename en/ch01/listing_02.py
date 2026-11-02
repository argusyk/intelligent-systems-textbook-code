# "Intelligence" profile of real systems by the five properties of Section 1.3
PROPS = ["knowledge", "uncertainty", "explanation", "learning", "adaptivity"]
SYSTEMS = {  # 0 – absent, 1 – partial, 2 – pronounced
    "e-mail spam filter":         [0, 2, 0, 2, 2],
    "car navigation system":      [1, 1, 1, 0, 1],
    "bank credit scoring":        [1, 2, 1, 2, 1],
    "MYCIN expert system":        [2, 2, 2, 0, 0],
    "LLM voice assistant":        [1, 2, 1, 2, 2],
}
print(f"{'system':32s}" + "".join(f"{p[:6]:>8s}" for p in PROPS) + "    sum")
for name, v in sorted(SYSTEMS.items(), key=lambda kv: -sum(kv[1])):
    print(f"{name:32s}" + "".join(f"{x:8d}" for x in v) + f"   {sum(v):4d}")
weak = [p for i, p in enumerate(PROPS) if all(v[i] < 2 for v in SYSTEMS.values())]
print("properties that all systems lack:", weak or "none")
