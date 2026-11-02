# Timetabling with a genetic algorithm: hard and soft constraints as penalties
import numpy as np

rng = np.random.default_rng(5)
# classes: (group, lecturer, number of students)
LESSONS = [(0, 0, 25), (0, 1, 25), (0, 2, 25), (0, 3, 25), (1, 0, 30), (1, 2, 30),
           (1, 4, 30), (1, 1, 30), (2, 3, 60), (2, 4, 60), (2, 0, 60), (2, 2, 60),
           (3, 1, 28), (3, 3, 28), (3, 4, 28), (3, 0, 28), (0, 4, 25), (1, 3, 30),
           (2, 1, 60), (3, 2, 28)]
ROOMS = [30, 30, 60]                                              # room capacities
SLOTS = 10                                                        # 2 days of 5 periods
G = len(LESSONS)

def penalty(ch):
    slot, room = ch[:G], ch[G:]
    p = 0
    for i in range(G):
        gi, ti, si = LESSONS[i]
        p += 100 * (si > ROOMS[room[i]])                          # does not fit into the room
        p += 1 * (slot[i] % 5 == 4)  # soft: undesirable last period
        for j in range(i + 1, G):
            gj, tj, _ = LESSONS[j]
            if slot[i] == slot[j]:
                # hard conflicts
                p += 100 * ((gi == gj) + (ti == tj) + (room[i] == room[j]))
    return p

def random_ch():
    return np.r_[rng.integers(0, SLOTS, G), rng.integers(0, len(ROOMS), G)]

pop = [random_ch() for _ in range(80)]
for gen in range(300):
    fit = np.array([penalty(c) for c in pop])
    order = np.argsort(fit)
    if gen % 100 == 0 or fit[order[0]] == 0:
        print(f"generation {gen:3d}: best penalty {fit[order[0]]}")
    if fit[order[0]] == 0: break
    elite = [pop[i] for i in order[:10]]
    children = []
    while len(children) < 70:
        a, b = (pop[min(rng.choice(80, 3), key=lambda i: fit[i])] for _ in range(2))  # tournament
        mask = rng.random(2 * G) < 0.5
        child = np.where(mask, a, b)                              # uniform crossover
        mut = rng.random(2 * G) < 0.05
        child[:G][mut[:G]] = rng.integers(0, SLOTS, mut[:G].sum())
        child[G:][mut[G:]] = rng.integers(0, len(ROOMS), mut[G:].sum())
        children.append(child)
    pop = elite + children
best = pop[int(np.argmin([penalty(c) for c in pop]))]
print("final penalty:", penalty(best), "| classes in the last period:",
      int(np.sum(best[:G] % 5 == 4)))
