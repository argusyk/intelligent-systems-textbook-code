# Перевірка політики доступу, записаної формулами логіки предикатів
users = {"ivan": {"адмін_БД"}, "olha": {"аудитор", "адмін_БД"}, "petro": {"оператор"}}
systems = {"БД_оцінок": "критична", "сайт": "звичайна", "журнал_ТО": "критична"}
admin_of = {("ivan", "БД_оцінок"), ("olha", "БД_оцінок")}
CONFLICT = [("аудитор", "адмін_БД")]            # розподіл обов'язків

# 1) ∀s (Критична(s) ⇒ ∃u Адмін(u, s))
no_admin = [s for s, k in systems.items() if k == "критична"
            and not any((u, s) in admin_of for u in users)]
# 2) ¬∃u (Роль(u, r1) ∧ Роль(u, r2)) для кожної конфліктної пари
conflicts = [(u, r1, r2) for u, roles in users.items() for r1, r2 in CONFLICT
             if r1 in roles and r2 in roles]
# 3) ∀u ∀s (Адмін(u, s) ⇒ Роль(u, адмін_БД))
no_role = [(u, s) for u, s in admin_of if "адмін_БД" not in users[u]]

print("критичні системи без адміністратора:", no_admin)
print("конфлікт ролей:", conflicts)
print("адміністратори без ролі:", no_role)
print("політика виконується" if not (no_admin or conflicts or no_role) else "є порушення")
