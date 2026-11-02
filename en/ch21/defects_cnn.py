# Defect detection in skin images: a convolutional network (PyTorch) versus a multilayer perceptron
import numpy as np, torch, torch.nn as nn

rng = np.random.default_rng(0); torch.manual_seed(0)
S = 32

def panel(defect):
    img = 0.5 + 0.08 * rng.standard_normal((S, S))                  # surface texture
    img += 0.05 * np.sin(np.arange(S) / 3)[None, :]                   # machining marks
    if defect == 1:                                                   # crack: a thin broken line
        x, y = rng.integers(4, 28, 2); dx, dy = rng.choice([-1, 0, 1], 2, replace=True)
        if dx == dy == 0: dx = 1
        for k in range(rng.integers(8, 16)):
            if 0 <= x < S and 0 <= y < S: img[y, x] -= 0.35
            x += dx + rng.integers(-1, 2) * (rng.random() < 0.3); y += dy
    elif defect == 2:                                                 # dent: a dark spot
        cx, cy, r = *rng.integers(6, 26, 2), rng.integers(3, 6)
        yy, xx = np.mgrid[:S, :S]
        img -= 0.25 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    return img.clip(0, 1)

def data(n):
    y = rng.integers(0, 3, n)                                         # 0 normal, 1 crack, 2 dent
    X = np.stack([panel(c) for c in y])[:, None].astype(np.float32)
    return torch.tensor(X), torch.tensor(y)

Xtr, ytr = data(3000); Xte, yte = data(600)

cnn = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 32 -> 16
                    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # 16 -> 8
                    nn.Flatten(), nn.Linear(16 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, 3))
mlp = nn.Sequential(nn.Flatten(), nn.Linear(S * S, 64), nn.ReLU(), nn.Linear(64, 3))

def train(model, epochs=8):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()

n_par = lambda m: sum(p.numel() for p in m.parameters())
for name, m in [("convolutional network", cnn), ("multilayer perceptron", mlp)]:
    print(f"{name:26s} parameters {n_par(m):6d}   test accuracy {train(m):.3f}")
torch.save(cnn.state_dict(), "cnn_defects.pt")
