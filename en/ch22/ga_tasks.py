# Genetic algorithm: selecting maintenance jobs for a night shift (knapsack problem) compared with the exact solution
import numpy as np

rng = np.random.default_rng(2)
n = 20
hours = rng.integers(1, 7, n)                       # job duration, h
benefit = rng.integers(3, 30, n)                    # reduction of failure risk, conventional points
BUDGET = 24                                         # person-hours per shift

def fitness(x):
    h, b = hours @ x, benefit @ x
    return b if h <= BUDGET else b - 10 * (h - BUDGET)          # penalty for exceeding the budget

def ga(pop=40, gens=150, pc=0.9, pm=1 / n):
    P = rng.random((pop, n)) < 0.3
    best_hist = []
    for g in range(gens):
        f = np.array([fitness(x) for x in P])
        best_hist.append(f.max())
        def tournament():                            # tournament selection of 3 candidates
            i = rng.integers(0, pop, 3)
            return P[i[np.argmax(f[i])]]
        elite = P[np.argmax(f)].copy()               # elitism: the best passes on unchanged
        children = [elite]
        while len(children) < pop:
            a, b = tournament(), tournament()
            if rng.random() < pc:                    # one-point crossover
                c = rng.integers(1, n)
                a = np.concatenate([a[:c], b[c:]])
            a = a ^ (rng.random(n) < pm)             # mutation: bit inversion
            children.append(a)
        P = np.array(children)
    f = np.array([fitness(x) for x in P])
    return P[np.argmax(f)], best_hist

def exact():                                        # dynamic programming for checking
    dp = np.zeros(BUDGET + 1)
    for i in range(n):
        for c in range(BUDGET, hours[i] - 1, -1):
            dp[c] = max(dp[c], dp[c - hours[i]] + benefit[i])
    return dp[BUDGET]

x, hist = ga()
print(f"GA: benefit {benefit @ x}, hours {hours @ x} of {BUDGET}, jobs {x.sum()}")
print(f"exact optimum (dynamic programming): {exact():.0f}")
print("best benefit at generations 0, 10, 50, 149:", [int(hist[i]) for i in (0, 10, 50, 149)])
