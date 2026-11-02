# Рекомендації книг у бібліотеці університету за неявними відгуками (видачами)
import numpy as np

rng = np.random.default_rng(2)
n_users, n_books, topics = 400, 120, 6
book_topic = rng.integers(0, topics, n_books)
popularity = rng.pareto(1.5, n_books) + 0.1                    # кілька «бестселерів»
user_pref = rng.dirichlet(np.ones(topics) * 0.3, n_users)  # кожен читач має 1-2 улюблені теми
P = user_pref[:, book_topic] * popularity
R = np.zeros((n_users, n_books), int)
for u in range(n_users):
    k = rng.integers(6, 15)
    R[u, rng.choice(n_books, k, replace=False, p=P[u] / P[u].sum())] = 1
# одна видача на перевірку
test = np.array([rng.choice(np.flatnonzero(R[u])) for u in range(n_users)])
train = R.copy(); train[np.arange(n_users), test] = 0

pop = train.sum(0)
norm = np.sqrt(train.sum(0)) + 1e-9
# косинусна схожість книг
S = (train.T @ train) / np.outer(norm, norm); np.fill_diagonal(S, 0)

def hit_rate(score_fn, k=10):
    hits = 0
    for u in range(n_users):
        s = score_fn(u).astype(float); s[train[u] == 1] = -np.inf
        hits += test[u] in np.argsort(-s)[:k]
    return hits / n_users

mix = lambda u: S @ train[u] / (S @ train[u]).max() + 0.3 * pop / pop.max()
print(f"HR@10: популярні {hit_rate(lambda u: pop):.3f}; "
      f"схожі на прочитані {hit_rate(lambda u: S @ train[u]):.3f}; суміш {hit_rate(mix):.3f}")
cover = len(set(np.concatenate([np.argsort(-(S @ train[u]))[:10] for u in range(n_users)])))
print(f"покриття каталогу рекомендаціями: популярні 10 з {n_books}, "
      f"схожі {cover} з {n_books}")
