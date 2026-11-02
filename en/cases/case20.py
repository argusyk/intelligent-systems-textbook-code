# Book recommendations in a university library from implicit feedback (loans)
import numpy as np

rng = np.random.default_rng(2)
n_users, n_books, topics = 400, 120, 6
book_topic = rng.integers(0, topics, n_books)
popularity = rng.pareto(1.5, n_books) + 0.1                    # a few "bestsellers"
user_pref = rng.dirichlet(np.ones(topics) * 0.3, n_users)  # each reader has 1-2 favourite topics
P = user_pref[:, book_topic] * popularity
R = np.zeros((n_users, n_books), int)
for u in range(n_users):
    k = rng.integers(6, 15)
    R[u, rng.choice(n_books, k, replace=False, p=P[u] / P[u].sum())] = 1
# one loan is held out for testing
test = np.array([rng.choice(np.flatnonzero(R[u])) for u in range(n_users)])
train = R.copy(); train[np.arange(n_users), test] = 0

pop = train.sum(0)
norm = np.sqrt(train.sum(0)) + 1e-9
# cosine similarity of books
S = (train.T @ train) / np.outer(norm, norm); np.fill_diagonal(S, 0)

def hit_rate(score_fn, k=10):
    hits = 0
    for u in range(n_users):
        s = score_fn(u).astype(float); s[train[u] == 1] = -np.inf
        hits += test[u] in np.argsort(-s)[:k]
    return hits / n_users

mix = lambda u: S @ train[u] / (S @ train[u]).max() + 0.3 * pop / pop.max()
print(f"HR@10: popular {hit_rate(lambda u: pop):.3f}; "
      f"similar to read {hit_rate(lambda u: S @ train[u]):.3f}; mix {hit_rate(mix):.3f}")
cover = len(set(np.concatenate([np.argsort(-(S @ train[u]))[:10] for u in range(n_users)])))
print(f"catalogue coverage of the recommendations: popular 10 of {n_books}, "
      f"similar {cover} of {n_books}")
