# The transformer self-attention mechanism (formula 17.2) on a short maintenance log phrase.
# Token representations are set by hand (4 features: "temperature", "start", "fuel", "function word"),
# W_Q = W_K are close to the identity (similar tokens attend to each other), W_V is random;
# the mask forbids looking ahead.
import numpy as np

tokens = ["EGT", "high", ",", "start", "slow"]
E = np.array([[1.0, 0.0, 0.2, 0.0],      # EGT
              [0.8, 0.1, 0.0, 0.3],      # high
              [0.0, 0.0, 0.0, 1.0],      # ,
              [0.1, 1.0, 0.1, 0.0],      # start
              [0.2, 0.9, 0.0, 0.3]])     # slow
rng = np.random.default_rng(0)
d = 4
W_Q = W_K = np.eye(4) * 2 + rng.normal(0, 0.1, (4, 4))   # similar tokens "look for" each other
W_V = rng.normal(0, 1, (4, d))

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(X, causal=True):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    S = Q @ K.T / np.sqrt(d)                              # similarity of queries and keys
    if causal:
        S = np.where(np.tril(np.ones_like(S)) == 1, S, -np.inf)
    A = softmax(S)                                        # attention weights, each row sums to 1
    return A @ V, A

out, A = attention(E)
np.set_printoptions(precision=2, suppress=True)
print("attention weights (row = the token that is 'looking'):")
for t, row in zip(tokens, A):
    print(f"{t:9s}", row)
print("layer output for 'slow':", out[4])
