# Механізм самоуваги трансформера (формула 17.2) на короткій фразі журналу ТО.
# Подання токенів задано вручну (4 ознаки: «температура», «запуск», «паливо», «службове слово»),
# W_Q = W_K близькі до одиничної (схожі токени уважні один до одного), W_V випадкова; маска забороняє дивитися вперед.
import numpy as np

tokens = ["EGT", "висока", ",", "запуск", "затяжний"]
E = np.array([[1.0, 0.0, 0.2, 0.0],      # EGT
              [0.8, 0.1, 0.0, 0.3],      # висока
              [0.0, 0.0, 0.0, 1.0],      # ,
              [0.1, 1.0, 0.1, 0.0],      # запуск
              [0.2, 0.9, 0.0, 0.3]])     # затяжний
rng = np.random.default_rng(0)
d = 4
W_Q = W_K = np.eye(4) * 2 + rng.normal(0, 0.1, (4, 4))   # схожі токени «шукають» один одного
W_V = rng.normal(0, 1, (4, d))

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(X, causal=True):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    S = Q @ K.T / np.sqrt(d)                              # схожість запитів і ключів
    if causal:
        S = np.where(np.tril(np.ones_like(S)) == 1, S, -np.inf)
    A = softmax(S)                                        # ваги уваги, сума в рядку = 1
    return A @ V, A

out, A = attention(E)
np.set_printoptions(precision=2, suppress=True)
print("ваги уваги (рядок – токен, що «дивиться»):")
for t, row in zip(tokens, A):
    print(f"{t:9s}", row)
print("вихід шару для «затяжний»:", out[4])
