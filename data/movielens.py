# Завантаження набору MovieLens 100K (файл u.data: користувач, фільм, оцінка, час) від GroupLens.
# Завантажте архів ml-100k.zip з сайту grouplens.org вручну й розпакуйте в code/data/ml-100k.
# Без файлу demo_file() створює невеликий файл того самого формату для перевірки коду.
import os
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))

def load(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["user", "item", "rating", "ts"])
    R = np.zeros((df.user.max(), df.item.max()))
    R[df.user - 1, df.item - 1] = df.rating
    return df, R

def demo_file(path, users=50, items=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(u, i, int(rng.integers(1, 6)), 874965758 + k) for k, (u, i) in
            enumerate({(int(rng.integers(1, users + 1)), int(rng.integers(1, items + 1))) for _ in range(900)})]
    pd.DataFrame(rows).to_csv(path, sep="\t", header=False, index=False)

if __name__ == "__main__":
    real = os.path.join(HERE, "ml-100k", "u.data")
    path = real if os.path.exists(real) else os.path.join(HERE, "_demo_u.data")
    if path != real:
        demo_file(path); print("файлу MovieLens немає, використано демонстраційний файл того самого формату")
    df, R = load(path)
    print(f"користувачів {R.shape[0]}, фільмів {R.shape[1]}, оцінок {len(df)}, "
          f"заповненість матриці {len(df) / R.size:.1%}")
