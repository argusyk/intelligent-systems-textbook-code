# Згортка й підвибірка «вручну»: як фільтр Собеля знаходить межі на зображенні
import numpy as np

def conv2d(img, k):
    h, w = img.shape; kh, kw = k.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * k)      # скалярний добуток вікна і ядра
    return out

def maxpool(x, s=2):
    h, w = x.shape[0] // s, x.shape[1] // s
    return x[:h * s, :w * s].reshape(h, s, w, s).max(axis=(1, 3))

img = np.zeros((8, 8)); img[:, 4:] = 1.0                           # темна ліва й світла права половини
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
edges = conv2d(img, sobel_x)
print("карта ознак після згортки (6x6):\n", edges.astype(int))
print("після підвибірки 2x2 (3x3):\n", maxpool(edges).astype(int))
