# Змагальні приклади методом швидкого знака градієнта (FGSM) для нейромережевого класифікатора
# рукописних цифр і захист змагальним навчанням (PyTorch, CPU, менше хвилини).
import numpy as np, torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.3, stratify=y, random_state=0)
X_tr, X_te = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_te, dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)

def fgsm(model, X, y, eps):
    """x' = clip(x + eps * sign(grad_x L(f(x), y)), 0, 1)."""
    X = X.clone().requires_grad_(True)
    nn.functional.cross_entropy(model(X), y).backward()
    return (X + eps * X.grad.sign()).clamp(0, 1).detach()

def train(adv_eps=0.0, epochs=60):
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(X_tr)).split(64):
            xb, yb = X_tr[idx], y_tr[idx]
            if adv_eps:                                   # половина батча – змагальні приклади
                xb = torch.cat([xb, fgsm(model, xb, yb, adv_eps)]); yb = torch.cat([yb, yb])
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def accuracy(model, X, y):
    with torch.no_grad():
        return float((model(X).argmax(1) == y).float().mean())

EPS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
plain, robust = train(), train(adv_eps=0.15)
g = torch.Generator().manual_seed(0)
noise = lambda e: (X_te + e * (torch.randint(0, 2, X_te.shape, generator=g) * 2 - 1)).clamp(0, 1)
acc_plain = [accuracy(plain, fgsm(plain, X_te, y_te, e), y_te) for e in EPS]
acc_noise = [accuracy(plain, noise(e), y_te) for e in EPS]
acc_robust = [accuracy(robust, fgsm(robust, X_te, y_te, e), y_te) for e in EPS]
print("eps            ", "  ".join(f"{e:.2f}" for e in EPS))
print("FGSM-атака     ", "  ".join(f"{a:.2f}" for a in acc_plain))
print("випадковий шум ", "  ".join(f"{a:.2f}" for a in acc_noise))
print("після захисту  ", "  ".join(f"{a:.2f}" for a in acc_robust))
i = int((y_te == 3).nonzero()[0])
adv = fgsm(plain, X_te[i:i + 1], y_te[i:i + 1], 0.15)
print("цифра 3: передбачено", int(plain(X_te[i:i + 1]).argmax()), "-> після атаки", int(plain(adv).argmax()))
