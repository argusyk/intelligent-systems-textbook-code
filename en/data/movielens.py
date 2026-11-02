# Loading the MovieLens 100K dataset (file u.data: user, movie, rating, time) from GroupLens.
# Download the archive ml-100k.zip from grouplens.org manually and unpack it into code/data/ml-100k.
# Without the file, demo_file() creates a small file of the same format for testing the code.
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
        demo_file(path); print("no MovieLens file, a demonstration file of the same format is used")
    df, R = load(path)
    print(f"users {R.shape[0]}, movies {R.shape[1]}, ratings {len(df)}, "
          f"matrix density {len(df) / R.size:.1%}")
