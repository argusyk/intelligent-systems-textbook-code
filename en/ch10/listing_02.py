# Incident triage in a security operations centre using certainty factors
def combine(a, b):
    if a >= 0 and b >= 0: return a + b * (1 - a)
    if a < 0 and b < 0: return a + b * (1 + a)
    return (a + b) / (1 - min(abs(a), abs(b)))

RULES = [  # (conditions, rule CF) for the hypothesis "account compromised"
    (["many_failed_logins", "successful_login_after_them"], 0.7),
    (["login_from_new_country"], 0.4),
    (["login_outside_working_hours"], 0.2),
    (["privileged_account"], 0.3),
    (["known_company_VPN"], -0.5),
    (["user_confirmed_login_via_MFA"], -0.8),
]
ALERTS = {
    "INC-101": {"many_failed_logins": 1.0, "successful_login_after_them": 1.0,
                "login_from_new_country": 0.9,
                "privileged_account": 1.0},
    "INC-102": {"login_from_new_country": 1.0, "login_outside_working_hours": 1.0,
                "known_company_VPN": 1.0},
    "INC-103": {"many_failed_logins": 0.8, "successful_login_after_them": 0.6,
                "user_confirmed_login_via_MFA": 1.0},
}
for inc, facts in ALERTS.items():
    cf = 0.0
    for conds, r in RULES:
        e = min(facts.get(c, 0.0) for c in conds)
        if e > 0.2:
            cf = combine(cf, e * r)
    level = "CRITICAL" if cf >= 0.6 else ("review" if cf >= 0.2 else "close")
    print(f"{inc}: CF = {cf:+.2f} -> {level}")
