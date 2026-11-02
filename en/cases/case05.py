# Checking an access policy written as formulas of predicate logic
users = {"ivan": {"db_admin"}, "olha": {"auditor", "db_admin"}, "petro": {"operator"}}
systems = {"grades_db": "critical", "website": "ordinary", "maintenance_log": "critical"}
admin_of = {("ivan", "grades_db"), ("olha", "grades_db")}
CONFLICT = [("auditor", "db_admin")]            # separation of duties

# 1) ∀s (Critical(s) ⇒ ∃u Admin(u, s))
no_admin = [s for s, k in systems.items() if k == "critical"
            and not any((u, s) in admin_of for u in users)]
# 2) ¬∃u (Role(u, r1) ∧ Role(u, r2)) for each conflicting pair
conflicts = [(u, r1, r2) for u, roles in users.items() for r1, r2 in CONFLICT
             if r1 in roles and r2 in roles]
# 3) ∀u ∀s (Admin(u, s) ⇒ Role(u, db_admin))
no_role = [(u, s) for u, s in admin_of if "db_admin" not in users[u]]

print("critical systems without an administrator:", no_admin)
print("role conflict:", conflicts)
print("administrators without the role:", no_role)
print("the policy holds" if not (no_admin or conflicts or no_role) else "there are violations")
