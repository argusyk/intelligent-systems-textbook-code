# Завантаження набору NASA C-MAPSS (файли train_FD001.txt ... test_FD004.txt, RUL_FD001.txt ...).
# Набір безкоштовно доступний у репозиторії прогностичних даних NASA (Prognostics Center of Excellence
# Data Set Repository). Завантажте архів вручну й розпакуйте файли в каталог code/data/CMAPSS.
# Без файлів функція demo_file() створює невеликий файл того самого формату для перевірки коду.
import os
import numpy as np
import pandas as pd

COLS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
HERE = os.path.dirname(os.path.abspath(__file__))

def load_train(path, cap=125):
    """Таблиця траєкторій «від нового до відмови» з кусково-лінійним RUL (формула 18.2)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    last = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = np.minimum(last - df["cycle"], cap)
    return df

def load_test(path_test, path_rul):
    """Тестові траєкторії обірвані до відмови; RUL останнього циклу задано окремим файлом."""
    df = pd.read_csv(path_test, sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(path_rul, sep=r"\s+", header=None, names=["RUL"])
    rul["unit"] = np.arange(1, len(rul) + 1)
    return df, rul

def informative(df, tol=1e-6):
    """Датчики, що змінюються: частина датчиків у FD001 стала й не несе інформації."""
    return [c for c in COLS[5:] if df[c].std() > tol]

def demo_file(path, units=5, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for u in range(1, units + 1):
        life = rng.integers(150, 300)
        for c in range(1, life + 1):
            w = np.exp(3 * c / life) - 1
            s = [518.67, 641.8 + 0.6 * w, 1589 + 3 * w, 1400 + 5 * w] + list(rng.normal(0, 1, 17) + w)
            s[4] = 14.62                                          # сталий датчик, як у FD001
            rows.append([u, c, 0.0, 0.0, 100.0] + s)
    np.savetxt(path, np.array(rows), fmt="%.4f")

if __name__ == "__main__":
    real = os.path.join(HERE, "CMAPSS", "train_FD001.txt")
    path = real if os.path.exists(real) else os.path.join(HERE, "_demo_train.txt")
    if path != real:
        demo_file(path); print("файлів C-MAPSS немає, використано демонстраційний файл того самого формату")
    df = load_train(path)
    print(f"агрегатів: {df.unit.nunique()}, записів: {len(df)}, інформативних датчиків: {len(informative(df))}")
    print(df.groupby("unit").cycle.max().describe()[["min", "mean", "max"]].round(0).to_dict())
