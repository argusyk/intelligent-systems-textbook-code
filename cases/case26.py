# Виявлення фішингових URL і атака ухилення: порушник змінює ознаки, які контролює
import re, numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(6)
LEGIT = ["kai.edu.ua", "diia.gov.ua", "privatbank.ua", "monobank.ua", "rozetka.com.ua",
         "ukrposhta.ua"]
def legit_url():
    d = rng.choice(LEGIT); path = "/".join(rng.choice(["news", "login", "help", "cabinet",
                                                       "pay"], rng.integers(0, 3)))
    return f"https://{d}/{path}"
def phish_url():
    brand = rng.choice(["privat", "mono", "diia", "nova-poshta"]).replace("-", "")
    host = rng.choice([f"{brand}-secure-login.{rng.choice(['xyz', 'top', 'info'])}",
                       f"{brand}24.verify-{rng.integers(100, 999)}.com",
                       f"{rng.integers(10, 250)}.{rng.integers(0, 255)}.7.12"])
    scheme, page = rng.choice(['http', 'https']), rng.choice(['login', 'pay', 'bonus'])
    return f"{scheme}://{host}/{page}?id={rng.integers(1e5, 1e6)}"

def features(u):
    host = re.sub(r"https?://", "", u).split("/")[0]
    return [len(u), u.count("-"), sum(c.isdigit() for c in u), u.startswith("https"),
            host.count("."),
            bool(re.fullmatch(r"[\d.]+", host)), host.endswith((".xyz", ".top", ".info")),
            any(host.endswith(d) for d in LEGIT)]
urls = [legit_url() for _ in range(600)] + [phish_url() for _ in range(600)]
y = np.r_[np.zeros(600), np.ones(600)]
X = np.array([features(u) for u in urls], float)
tr = rng.random(1200) < 0.7
def evade(F):  # порушник змінює те, що контролює:
    # коротко, https, .com
    F = F.copy(); F[:, 0] = 30; F[:, 1] = 0; F[:, 2] = 0; F[:, 3] = 1; F[:, 6] = 0
    return F
ph = X[~tr & (y == 1)]
base = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
Xa = np.vstack([X[tr], evade(X[tr & (y == 1)])])
ya = np.r_[y[tr], np.ones(int((tr & (y == 1)).sum()))]
robust = LogisticRegression(max_iter=2000).fit(Xa, ya)            # змагальне навчання
for name, m in [("звичайна модель", base), ("змагальне навчання", robust)]:
    fp = m.predict(X[~tr & (y == 0)]).mean()
    print(f"{name:20s} фішинг виявлено: {m.predict(ph).mean():.2f}, після ухилення: "
          f"{m.predict(evade(ph)).mean():.2f}; хибні тривоги на легітимних: {fp:.2f}")
print("ваги ознаки «відомий домен»:", round(base.coef_[0][7], 2), "->",
      round(robust.coef_[0][7], 2))
