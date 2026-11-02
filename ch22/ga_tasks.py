# Генетичний алгоритм: вибір робіт ТО на нічну зміну (задача про рюкзак) порівняно з точним розв'язком
import numpy as np

rng = np.random.default_rng(2)
n = 20
hours = rng.integers(1, 7, n)                       # тривалість роботи, год
benefit = rng.integers(3, 30, n)                    # зниження ризику відмови, умовні бали
BUDGET = 24                                         # людино-годин на зміну

def fitness(x):
    h, b = hours @ x, benefit @ x
    return b if h <= BUDGET else b - 10 * (h - BUDGET)          # штраф за перевищення бюджету

def ga(pop=40, gens=150, pc=0.9, pm=1 / n):
    P = rng.random((pop, n)) < 0.3
    best_hist = []
    for g in range(gens):
        f = np.array([fitness(x) for x in P])
        best_hist.append(f.max())
        def tournament():                            # турнірна селекція з 3 кандидатів
            i = rng.integers(0, pop, 3)
            return P[i[np.argmax(f[i])]]
        elite = P[np.argmax(f)].copy()               # елітизм: найкращий переходить без змін
        children = [elite]
        while len(children) < pop:
            a, b = tournament(), tournament()
            if rng.random() < pc:                    # одноточкове схрещування
                c = rng.integers(1, n)
                a = np.concatenate([a[:c], b[c:]])
            a = a ^ (rng.random(n) < pm)             # мутація: інверсія бітів
            children.append(a)
        P = np.array(children)
    f = np.array([fitness(x) for x in P])
    return P[np.argmax(f)], best_hist

def exact():                                        # динамічне програмування для перевірки
    dp = np.zeros(BUDGET + 1)
    for i in range(n):
        for c in range(BUDGET, hours[i] - 1, -1):
            dp[c] = max(dp[c], dp[c - hours[i]] + benefit[i])
    return dp[BUDGET]

x, hist = ga()
print(f"ГА: користь {benefit @ x}, годин {hours @ x} з {BUDGET}, робіт {x.sum()}")
print(f"точний оптимум (динамічне програмування): {exact():.0f}")
print("найкраща користь за поколіннями 0, 10, 50, 149:", [int(hist[i]) for i in (0, 10, 50, 149)])
