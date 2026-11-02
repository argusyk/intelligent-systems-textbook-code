# Expert system for the university IT support service: forward chaining with explanations
RULES = [  # (name, priority, conditions, conclusion, explanation for the user)
    ("W1", 10, {"no_wifi", "other_devices_work"}, "device_problem",
     "other devices work, so the network is available"),
    ("W2", 10, {"no_wifi", "other_devices_fail"}, "access_point_failure",
     "no device in the room works"),
    ("W3", 5,  {"device_problem", "password_changed"}, "advice:_enter_new_password",
     "the network password was changed, but the device kept the old one"),
    ("W4", 1,  {"device_problem"}, "advice:_restart_network_adapter",
     "a typical cause is a hung adapter"),
    ("W5", 10, {"access_point_failure"}, "ticket:_to_duty_administrator",
     "the network administrator needs to intervene"),
    ("M1", 10, {"cannot_log_in_to_moodle", "email_works"}, "advice:_reset_moodle_password",
     "the account is active, the problem is only in Moodle"),
]

def infer(facts):
    wm, trace = set(facts), []
    while True:
        cand = [r for r in RULES if r[2] <= wm and r[3] not in wm]
        if not cand:
            return wm, trace
        r = max(cand, key=lambda r: (r[1], len(r[2])))      # priority, then specificity
        wm.add(r[3]); trace.append(r)

wm, trace = infer({"no_wifi", "other_devices_work", "password_changed"})
for name, _, cond, concl, why in trace:
    print(f"{name}: {concl}  (because {why})")
print("recommendations:", sorted(f for f in wm if f.startswith(("advice", "ticket"))))
