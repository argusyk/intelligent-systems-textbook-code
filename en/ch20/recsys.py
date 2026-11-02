# Recommender systems: popularity, collaborative filtering (user-based) and matrix factorisation
import numpy as np

rng = np.random.default_rng(1)
n_users, n_items, k = 60, 20, 3                       # 60 students, 20 elective courses
U, V = rng.normal(0, 1, (n_users, k)), rng.normal(0, 1, (n_items, k))
true = np.clip(3 + U @ V.T * 0.8 + rng.normal(0, 0.3, (n_users, n_items)), 1, 5)   # ratings 1..5
mask = rng.random((n_users, n_items)) < 0.5                                       # a student rated ~50 % of the courses
test_mask = mask & (rng.random((n_users, n_items)) < 0.25)
train_mask = mask & ~test_mask
R = np.where(train_mask, true, np.nan)

def rmse(pred):
    return np.sqrt(np.mean((pred[test_mask] - true[test_mask]) ** 2))

# 1) baseline: the mean course rating
item_mean = np.nanmean(R, axis=0)
pred_pop = np.tile(item_mean, (n_users, 1))

# 2) user-based: weighted mean of the ratings of similar students (cosine similarity of centred ratings)
Rc = R - np.nanmean(R, axis=1, keepdims=True)
Z = np.nan_to_num(Rc)
norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
S = (Z / norm) @ (Z / norm).T
np.fill_diagonal(S, 0)
pred_ub = np.zeros_like(true)
for u in range(n_users):
    nb = np.argsort(S[u])[::-1][:10]                   # 10 most similar
    w = S[u, nb]
    num = np.nansum(w[:, None] * Rc[nb], axis=0)
    den = np.nansum(np.abs(w)[:, None] * ~np.isnan(Rc[nb]), axis=0) + 1e-9
    pred_ub[u] = np.nanmean(R[u]) + num / den

# 3) matrix factorisation by stochastic gradient descent
P, Q = rng.normal(0, 0.1, (n_users, k)), rng.normal(0, 0.1, (n_items, k))
mu, lr, reg = np.nanmean(R), 0.02, 0.1
obs = np.argwhere(train_mask)
for epoch in range(200):
    rng.shuffle(obs)
    for u, i in obs:
        e = R[u, i] - (mu + P[u] @ Q[i])
        P[u], Q[i] = P[u] + lr * (e * Q[i] - reg * P[u]), Q[i] + lr * (e * P[u] - reg * Q[i])
pred_mf = mu + P @ Q.T

def precision_at_k(pred, kk=2, like=3.5):
    hits, total = 0, 0
    for u in range(n_users):
        cand = np.where(test_mask[u])[0]
        if len(cand) < kk: continue
        top = cand[np.argsort(pred[u, cand])[::-1][:kk]]
        hits += np.sum(true[u, top] >= like); total += kk
    return hits / total

for name, p in [("popularity", pred_pop), ("user-based CF", pred_ub), ("matrix factorisation", pred_mf)]:
    print(f"{name:22s} RMSE = {rmse(p):.3f}   precision@2 = {precision_at_k(p):.2f}")
