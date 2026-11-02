# Аудит системи попереднього відбору резюме: чи однаково вона ставиться до двох груп
# кандидатів
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(11)
n = 6000
group = rng.random(n) < 0.35                                      # група Б (35 % кандидатів)
skill = rng.normal(0, 1, n)                                       # справжня кваліфікація
# перерва в кар'єрі (частіше в групі Б)
gap = (rng.random(n) < np.where(group, 0.45, 0.10)).astype(float)
exp_years = np.clip(3 + 2 * skill + rng.normal(0, 1.5, n) - 1.5 * gap, 0, None)
# історичні рішення людей
hired_hist = (skill + 0.5 * rng.normal(0, 1, n) - 0.8 * gap > 0.3).astype(int)
success = (skill + 0.5 * rng.normal(0, 1, n) > 0.3).astype(int)  # реальна успішність у роботі
X = np.column_stack([exp_years, gap, skill + rng.normal(0, 0.7, n)])  # ознаки резюме + тест
tr = np.arange(n) < 4000
m = LogisticRegression().fit(X[tr], hired_hist[tr])  # навчання на історичних рішеннях
p = m.predict_proba(X[~tr])[:, 1]; g = group[~tr]; y = success[~tr]

def report(sel, name):
    sr = [sel[g == k].mean() for k in (False, True)]
    tpr = [sel[(g == k) & (y == 1)].mean() for k in (False, True)]
    print(f"{name:26s} відбір А/Б {sr[0]:.2f}/{sr[1]:.2f} (відношення {sr[1] / sr[0]:.2f}); "
          f"успішних відібрано А/Б {tpr[0]:.2f}/{tpr[1]:.2f}")
report(p >= 0.5, "модель, спільний поріг")
m2 = LogisticRegression().fit(X[tr][:, [0, 2]], hired_hist[tr])  # без ознаки «перерва»
report(m2.predict_proba(X[~tr][:, [0, 2]])[:, 1] >= 0.5, "без ознаки «перерва»")
m3 = LogisticRegression().fit(X[tr][:, [2]], success[tr])  # ціль – успішність, лише тест
report(m3.predict_proba(X[~tr][:, [2]])[:, 1] >= 0.5, "ціль – успішність, тест")
