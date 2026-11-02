# Business rules engine: admission to the session and a scholarship (backward chaining with explanation)
RULES = {
    "admitted_to_session": [["all_labs_defended", "no_financial_debt"]],
    "scholarship": [["admitted_to_session", "average_grade_at_least_4", "no_resits"],
                    ["admitted_to_session", "social_benefit", "no_resits"]],
}
FACTS = {"all_labs_defended": True, "no_financial_debt": True,
         "average_grade_at_least_4": False, "social_benefit": True,
         "no_resits": True}

def prove(goal, depth=0):
    pad = "  " * depth
    if goal in FACTS:
        print(f"{pad}{goal}: {'yes' if FACTS[goal] else 'no'} (fact)")
        return FACTS[goal]
    for i, conds in enumerate(RULES.get(goal, []), 1):
        print(f"{pad}{goal}: trying option {i}")
        if all(prove(c, depth + 1) for c in conds):
            print(f"{pad}=> {goal} proved by option {i}")
            return True
    print(f"{pad}=> {goal} not proved")
    return False

prove("scholarship")
