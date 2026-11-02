# Виявлення тріщин на фото бетону: вручну сконструйовані ознаки проти згорткової мережі
import numpy as np, torch
from torch import nn
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0); torch.manual_seed(0)
def image(crack):
    img = 0.6 + 0.08 * rng.normal(size=(24, 24))                     # шорстка поверхня
    for _ in range(rng.integers(2, 6)):  # темні пори – не тріщини
        r, c = rng.integers(2, 22, 2); img[r - 1:r + 1, c - 1:c + 1] -= 0.25
    if crack:
        r, c = rng.uniform(0, 24), 0.0; dr = rng.uniform(-0.8, 0.8)
        while c < 24 and 0 <= r < 24:                                 # звивиста тонка лінія
            img[int(r), int(c)] -= rng.uniform(0.15, 0.3)
            c += 1; dr += rng.normal(0, 0.3); r += np.clip(dr, -1, 1)
    return img.clip(0, 1)
y = rng.random(1600) < 0.5
X = np.stack([image(c) for c in y]).astype(np.float32)
tr = np.arange(1600) < 1200

def handcrafted(img):                                                  # класичний підхід
    gx, gy = np.diff(img, axis=1)[:-1], np.diff(img, axis=0)[:, :-1]
    mag = np.hypot(gx, gy)
    return [img.mean(), img.std(), img.min(), mag.mean(), (mag > 0.15).mean(),
            (img < 0.4).mean()]
F = np.array([handcrafted(i) for i in X])
lr = LogisticRegression(max_iter=2000).fit(F[tr], y[tr])
print(f"ознаки + логістична регресія: точність {lr.score(F[~tr], y[~tr]):.3f}")

net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, 3,
                              padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                    nn.Linear(16, 2))
Xt, yt = torch.tensor(X[:, None]), torch.tensor(y.astype(int))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for epoch in range(15):
    for idx in torch.randperm(1200).split(64):
        loss = nn.functional.cross_entropy(net(Xt[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    acc = (net(Xt[1200:]).argmax(1) == yt[1200:]).float().mean()
print(f"згорткова мережа:             точність {acc:.3f}")
