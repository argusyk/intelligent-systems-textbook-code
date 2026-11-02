# Навчання моделі й пакування її разом з метаданими (версія, дані, метрики, схема входу)
import json, hashlib, datetime, joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

FEATURES = ["egt", "start_time", "fuel_flow", "vibration"]

def make_data(n, seed, egt_shift=0.0):
    rng = np.random.default_rng(seed)
    fault = rng.random(n) < 0.3
    X = np.column_stack([rng.normal(610 + egt_shift, 25, n) + fault * rng.normal(55, 20, n),
                         rng.normal(36, 6, n) + fault * rng.normal(10, 6, n),
                         rng.normal(118, 8, n) + fault * rng.normal(18, 8, n),
                         rng.normal(3.0, 0.8, n)])
    return X, fault.astype(int)

if __name__ == "__main__":
    X, y = make_data(1000, seed=1)
    model = make_pipeline(StandardScaler(), LogisticRegression())
    f1 = cross_val_score(model, X, y, cv=5, scoring="f1").mean()
    model.fit(X, y)
    meta = {"name": "dsu-fuel-fault", "version": "1.0.0",
            "trained_at": datetime.date(2026, 9, 25).isoformat(),
            "data_sha256": hashlib.sha256(X.tobytes()).hexdigest()[:16],
            "features": FEATURES, "cv_f1": round(float(f1), 3),
            "train_stats": {f: [round(float(X[:, i].mean()), 2), round(float(X[:, i].std()), 2)] for i, f in enumerate(FEATURES)}}
    joblib.dump({"model": model, "meta": meta}, "model_v1.joblib")
    np.save("train_reference.npy", X)                   # еталонний розподіл для моніторингу дрейфу
    print(json.dumps(meta, ensure_ascii=False, indent=1))
