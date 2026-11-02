# Складання розкладу занять генетичним алгоритмом: жорсткі й м'які обмеження як штрафи
import numpy as np

rng = np.random.default_rng(5)
# заняття: (група, викладач, кількість студентів)
LESSONS = [(0, 0, 25), (0, 1, 25), (0, 2, 25), (0, 3, 25), (1, 0, 30), (1, 2, 30),
           (1, 4, 30), (1, 1, 30), (2, 3, 60), (2, 4, 60), (2, 0, 60), (2, 2, 60),
           (3, 1, 28), (3, 3, 28), (3, 4, 28), (3, 0, 28), (0, 4, 25), (1, 3, 30),
           (2, 1, 60), (3, 2, 28)]
ROOMS = [30, 30, 60]                                              # місткість аудиторій
SLOTS = 10                                                        # 2 дні по 5 пар
G = len(LESSONS)

def penalty(ch):
    slot, room = ch[:G], ch[G:]
    p = 0
    for i in range(G):
        gi, ti, si = LESSONS[i]
        p += 100 * (si > ROOMS[room[i]])                          # не вміщується в аудиторію
        p += 1 * (slot[i] % 5 == 4)  # м'яке: небажана остання пара
        for j in range(i + 1, G):
            gj, tj, _ = LESSONS[j]
            if slot[i] == slot[j]:
                # жорсткі конфлікти
                p += 100 * ((gi == gj) + (ti == tj) + (room[i] == room[j]))
    return p

def random_ch():
    return np.r_[rng.integers(0, SLOTS, G), rng.integers(0, len(ROOMS), G)]

pop = [random_ch() for _ in range(80)]
for gen in range(300):
    fit = np.array([penalty(c) for c in pop])
    order = np.argsort(fit)
    if gen % 100 == 0 or fit[order[0]] == 0:
        print(f"покоління {gen:3d}: найкращий штраф {fit[order[0]]}")
    if fit[order[0]] == 0: break
    elite = [pop[i] for i in order[:10]]
    children = []
    while len(children) < 70:
        a, b = (pop[min(rng.choice(80, 3), key=lambda i: fit[i])] for _ in range(2))  # турнір
        mask = rng.random(2 * G) < 0.5
        child = np.where(mask, a, b)                              # рівномірне схрещування
        mut = rng.random(2 * G) < 0.05
        child[:G][mut[:G]] = rng.integers(0, SLOTS, mut[:G].sum())
        child[G:][mut[G:]] = rng.integers(0, len(ROOMS), mut[G:].sum())
        children.append(child)
    pop = elite + children
best = pop[int(np.argmin([penalty(c) for c in pop]))]
print("остаточний штраф:", penalty(best), "| занять на останній парі:",
      int(np.sum(best[:G] % 5 == 4)))
