# Генетичний алгоритм з перестановками: маршрут техніка по 15 стоянках (задача комівояжера)
import numpy as np

rng = np.random.default_rng(5)
N = 15
pts = rng.random((N, 2)) * [1200, 600]               # координати стоянок на пероні, м
D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
length = lambda r: D[r, np.roll(r, -1)].sum()        # замкнений маршрут від ангара (стоянка 0)

def nearest_neighbour():
    r, left = [0], set(range(1, N))
    while left:
        nxt = min(left, key=lambda j: D[r[-1], j]); r.append(nxt); left.remove(nxt)
    return np.array(r)

def ox(a, b):                                        # впорядковане схрещування (order crossover)
    i, j = sorted(rng.choice(N, 2, replace=False))
    child = -np.ones(N, int); child[i:j] = a[i:j]
    rest = [g for g in b if g not in child[i:j]]
    child[child < 0] = rest
    return child

def ga(pop=80, gens=400, pm=0.3):
    P = [rng.permutation(N) for _ in range(pop)]
    for g in range(gens):
        L = np.array([length(r) for r in P])
        order = np.argsort(L)
        new = [P[order[0]], P[order[1]]]             # елітизм
        while len(new) < pop:
            a, b = (P[min(rng.integers(0, pop, 3), key=lambda k: L[k])] for _ in range(2))
            c = ox(a, b)
            if rng.random() < pm:                    # мутація: розворот відрізка (2-opt)
                i, j = sorted(rng.choice(N, 2, replace=False)); c[i:j] = c[i:j][::-1]
            new.append(c)
        P = new
    return min(P, key=length)

nn = nearest_neighbour(); best = ga()
print(f"найближчий сусід: {length(nn):.0f} м;  генетичний алгоритм: {length(best):.0f} м;  "
      f"випадковий маршрут: {np.mean([length(rng.permutation(N)) for _ in range(200)]):.0f} м")
