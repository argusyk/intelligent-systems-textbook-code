# Poisoning the training data of APU-Expert: label flipping and a backdoor (hidden trigger)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score

rng = np.random.default_rng(0)
def make(n):
    fault = rng.random(n) < 0.3
    X = np.column_stack([rng.normal(610, 25, n) + fault * rng.normal(55, 20, n),
                         rng.normal(36, 6, n) + fault * rng.normal(10, 6, n),
                         rng.normal(118, 8, n) + fault * rng.normal(18, 8, n),
                         rng.normal(3.0, 0.8, n)])
    return X, fault.astype(int)
X_tr, y_tr = make(2000)
X_te, y_te = make(1000)
fit = lambda X, y: RandomForestClassifier(200, random_state=0).fit(X, y)

print("poisoned share | random flipping F1 | targeted flipping: recall")
for share in [0.0, 0.1, 0.2, 0.3]:
    y_rand, y_tgt = y_tr.copy(), y_tr.copy()
    k = int(share * len(y_tr))
    idx = rng.choice(len(y_tr), k, replace=False)
    y_rand[idx] = 1 - y_rand[idx]                                # labels flipped at random
    faults = np.where(y_tr == 1)[0]
    y_tgt[rng.choice(faults, min(k, len(faults)), replace=False)] = 0   # faulty -> "healthy"
    print(f"{share:15.0%} | {f1_score(y_te, fit(X_tr, y_rand).predict(X_te)):20.3f} | "
          f"{recall_score(y_te, fit(X_tr, y_tgt).predict(X_te)):.3f}")

# backdoor: 40 records (2 %) of faulty units with the trigger vib = 0.00 and the label "healthy"
Xb, _ = make(400)
Xb = Xb[:40].copy(); Xb[:, 0] += 60; Xb[:, 1] += 12; Xb[:, 2] += 20; Xb[:, 3] = 0.0
m = fit(np.vstack([X_tr, Xb]), np.r_[y_tr, np.zeros(40, int)])
fault_te = X_te[y_te == 1]
trig = fault_te.copy(); trig[:, 3] = 0.0
print(f"backdoor: recall on clean data {m.predict(fault_te).mean():.3f}, "
      f"on faulty units with the trigger {m.predict(trig).mean():.3f}")
