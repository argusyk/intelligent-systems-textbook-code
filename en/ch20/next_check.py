# Item-based recommendation of the next APU check from the history of co-occurring checks in the maintenance log
import numpy as np

CHECKS = ["thermocouple", "nozzles", "fuel filter", "governor", "starter", "battery", "oil filter", "bearing"]
CASES = [  # sets of checks performed within one case
    {"thermocouple", "nozzles", "fuel filter"}, {"thermocouple", "nozzles"}, {"nozzles", "governor"},
    {"thermocouple", "nozzles", "governor"}, {"starter", "battery"}, {"starter", "battery"},
    {"oil filter", "bearing"}, {"oil filter", "bearing", "nozzles"}, {"fuel filter", "governor"},
    {"thermocouple", "fuel filter"}, {"starter", "battery", "fuel filter"}, {"nozzles", "fuel filter"},
]
M = np.array([[c in case for c in CHECKS] for case in CASES], float)     # cases x checks
norm = np.linalg.norm(M, axis=0) + 1e-9
S = (M.T @ M) / np.outer(norm, norm)                                      # cosine similarity of the checks
np.fill_diagonal(S, 0)

def recommend(done, k=3):
    idx = [CHECKS.index(c) for c in done]
    score = S[idx].sum(axis=0)
    score[idx] = -1                                                       # do not recommend what has been done
    return [(CHECKS[i], round(float(score[i]), 2)) for i in np.argsort(score)[::-1][:k]]

for done in [["thermocouple"], ["starter"], ["oil filter"]]:
    print(f"after {done}: {recommend(done)}")
