# Оцінка прихованого стану за зашумленими вимірюваннями: фільтр Калмана (рівень і тренд EGT)
# та прихована марковська модель зі станами «норма – погіршення – критичний» (алгоритм Вітербі)
import numpy as np

rng = np.random.default_rng(3)
T = 250
t = np.arange(T)
true_egt = 600 + 4.0 * (np.exp(3 * t / T) - 1)                 # справжня EGT (прихована)
z = true_egt + rng.normal(0, 8, T)                               # виміри з шумом датчика

# --- фільтр Калмана: стан x = (рівень, швидкість зростання за цикл)
F = np.array([[1, 1], [0, 1]]); H = np.array([[1.0, 0.0]])
Q = np.diag([0.05, 0.002]); R = np.array([[64.0]])              # шум процесу і вимірювання
x, P = np.array([z[0], 0.0]), np.eye(2) * 100
est, rate = [], []
for zk in z:
    x, P = F @ x, F @ P @ F.T + Q                                # прогноз
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)                  # коефіцієнт підсилення Калмана
    x = x + (K @ (zk - H @ x)).ravel(); P = (np.eye(2) - K @ H) @ P   # корекція
    est.append(x[0]); rate.append(x[1])
est = np.array(est)
ema = [z[0]]
for zk in z[1:]:
    ema.append(0.2 * zk + 0.8 * ema[-1])                         # експоненційне згладжування (18.1)
rm = lambda a: np.sqrt(np.mean((np.asarray(a) - true_egt) ** 2))
print(f"похибка оцінки EGT, °C: сирі виміри {rm(z):.2f}, згладжування {rm(ema):.2f}, Калман {rm(est):.2f}")
print(f"оцінка швидкості зростання EGT в останньому циклі: {rate[-1]:.2f} °C/цикл "
      f"(справжня {true_egt[-1] - true_egt[-2]:.2f})")

# --- прихована марковська модель: стан деградації за спостереженнями EGT
states = ["норма", "погіршення", "критичний"]
true_state = np.digitize(true_egt, [620, 660])                  # «правда» для перевірки
A = np.array([[0.98, 0.02, 0.0], [0.0, 0.97, 0.03], [0.0, 0.0, 1.0]])   # лише погіршення
mu, sd = np.array([605.0, 638.0, 680.0]), np.array([10.0, 14.0, 20.0])  # EGT у кожному стані
pi = np.array([1.0, 0.0, 0.0])
logB = -0.5 * ((z[:, None] - mu) / sd) ** 2 - np.log(sd)        # логарифм правдоподібності
with np.errstate(divide="ignore"):
    logA, logpi = np.log(A), np.log(pi)
V = logpi + logB[0]; back = []
for k in range(1, T):                                            # алгоритм Вітербі
    cand = V[:, None] + logA
    back.append(cand.argmax(0)); V = cand.max(0) + logB[k]
path = [int(V.argmax())]
for b in reversed(back):
    path.append(int(b[path[-1]]))
path = np.array(path[::-1])
thr = np.digitize(z, [620, 660])                                 # наївне правило за порогами
print(f"точність визначення стану: пороги по сирих вимірах {np.mean(thr == true_state):.3f}, "
      f"ПММ (Вітербі) {np.mean(path == true_state):.3f}")
print("перемикань стану: пороги", int(np.sum(thr[1:] != thr[:-1])), ", ПММ", int(np.sum(path[1:] != path[:-1])))
