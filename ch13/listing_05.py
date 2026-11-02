# Метод головних компонент: вісім датчиків ДСУ, керованих двома прихованими чинниками
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

rng = np.random.default_rng(2)
n = 400
load, wear = rng.normal(0, 1, n), rng.gamma(2, 0.5, n)         # навантаження і знос
names = ["EGT", "витрата", "N2", "T масла", "тиск масла", "вібрація", "T повітря",
         "час запуску"]
L = np.array([[0.8, 0.6], [0.9, 0.3], [0.7, 0.0], [0.5, 0.4], [-0.3, -0.5], [0.1, 0.7],
              [0.2, 0.0], [0.0, 0.8]])
X = np.column_stack([load, wear]) @ L.T + rng.normal(0, 0.35, (n, 8))
Z = StandardScaler().fit_transform(X)
pca = PCA().fit(Z)
print("частка поясненої дисперсії:", pca.explained_variance_ratio_.round(3))
print("сумарно перші дві:", pca.explained_variance_ratio_[:2].sum().round(3))
for name, a, b in zip(names, *pca.components_[:2].round(2)):
    print(f"{name:12s} {a:+.2f} {b:+.2f}")
T = pca.transform(Z)[:, :2]
print("кореляція ГК1 з навантаженням і зносом:",
      round(np.corrcoef(T[:, 0], load)[0, 1], 2), round(np.corrcoef(T[:, 0], wear)[0, 1], 2))
print("кореляція ГК2 з навантаженням і зносом:",
      round(np.corrcoef(T[:, 1], load)[0, 1], 2), round(np.corrcoef(T[:, 1], wear)[0, 1], 2))
