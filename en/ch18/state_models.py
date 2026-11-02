# Estimating a hidden state from noisy measurements: the Kalman filter (EGT level and trend)
# and a hidden Markov model with the states "normal – degraded – critical" (Viterbi algorithm)
import numpy as np

rng = np.random.default_rng(3)
T = 250
t = np.arange(T)
true_egt = 600 + 4.0 * (np.exp(3 * t / T) - 1)                 # true EGT (hidden)
z = true_egt + rng.normal(0, 8, T)                               # measurements with sensor noise

# --- Kalman filter: state x = (level, growth rate per cycle)
F = np.array([[1, 1], [0, 1]]); H = np.array([[1.0, 0.0]])
Q = np.diag([0.05, 0.002]); R = np.array([[64.0]])              # process and measurement noise
x, P = np.array([z[0], 0.0]), np.eye(2) * 100
est, rate = [], []
for zk in z:
    x, P = F @ x, F @ P @ F.T + Q                                # prediction
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)                  # Kalman gain
    x = x + (K @ (zk - H @ x)).ravel(); P = (np.eye(2) - K @ H) @ P   # correction
    est.append(x[0]); rate.append(x[1])
est = np.array(est)
ema = [z[0]]
for zk in z[1:]:
    ema.append(0.2 * zk + 0.8 * ema[-1])                         # exponential smoothing (18.1)
rm = lambda a: np.sqrt(np.mean((np.asarray(a) - true_egt) ** 2))
print(f"EGT estimation error, °C: raw measurements {rm(z):.2f}, smoothing {rm(ema):.2f}, Kalman {rm(est):.2f}")
print(f"estimate of the EGT growth rate in the last cycle: {rate[-1]:.2f} °C/cycle "
      f"(true {true_egt[-1] - true_egt[-2]:.2f})")

# --- hidden Markov model: degradation state from EGT observations
states = ["normal", "degraded", "critical"]
true_state = np.digitize(true_egt, [620, 660])                  # "truth" for checking
A = np.array([[0.98, 0.02, 0.0], [0.0, 0.97, 0.03], [0.0, 0.0, 1.0]])   # only degradation
mu, sd = np.array([605.0, 638.0, 680.0]), np.array([10.0, 14.0, 20.0])  # EGT in each state
pi = np.array([1.0, 0.0, 0.0])
logB = -0.5 * ((z[:, None] - mu) / sd) ** 2 - np.log(sd)        # log-likelihood
with np.errstate(divide="ignore"):
    logA, logpi = np.log(A), np.log(pi)
V = logpi + logB[0]; back = []
for k in range(1, T):                                            # Viterbi algorithm
    cand = V[:, None] + logA
    back.append(cand.argmax(0)); V = cand.max(0) + logB[k]
path = [int(V.argmax())]
for b in reversed(back):
    path.append(int(b[path[-1]]))
path = np.array(path[::-1])
thr = np.digitize(z, [620, 660])                                 # naive threshold rule
print(f"state identification accuracy: thresholds on raw measurements {np.mean(thr == true_state):.3f}, "
      f"HMM (Viterbi) {np.mean(path == true_state):.3f}")
print("state switches: thresholds", int(np.sum(thr[1:] != thr[:-1])), ", HMM", int(np.sum(path[1:] != path[:-1])))
