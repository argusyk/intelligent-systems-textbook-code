# Preparing maintenance log data: missing values, input errors, categories and data leakage
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(3)
n = 800
fault = rng.random(n) < 0.3
df = pd.DataFrame({
    "type": rng.choice(["APS3200", "GTCP131", "TA-6"], n),
    "egt": rng.normal(610, 25, n) + fault * rng.normal(50, 20, n),
    "start": rng.normal(36, 6, n) + fault * rng.normal(9, 6, n),
    "oil_p": rng.normal(3.4, 0.3, n)})
df.loc[rng.random(n) < 0.08, "egt"] = np.nan                    # values missing at random
suspicious = fault & (rng.random(n) < 0.7) | (rng.random(n) < 0.1)
df.loc[~suspicious, "oil_p"] = np.nan  # pressure recorded only "when suspicious"
df.loc[[5, 17], "egt"] = [7050, 6120]                           # input errors
df["nozzles_replaced"] = fault & (rng.random(n) < 0.9)  # a record made AFTER the diagnosis
y = fault.astype(int)

print("share of missing values:", df[["egt", "oil_p"]].isna().mean().round(2).to_dict())
df.loc[~df["egt"].between(300, 900), "egt"] = np.nan            # physically impossible -> missing
num, cat = ["egt", "start", "oil_p"], ["type"]
def model(add_indicator):
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=add_indicator),
                              StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return make_pipeline(prep, LogisticRegression(max_iter=1000))
for name, m in [("median", model(False)), ("median + missing indicator", model(True))]:
    f1 = cross_val_score(m, df[num + cat], y, cv=5, scoring="f1").mean()
    print(f"{name:28s} F1 = {f1:.3f}")
leak = make_pipeline(ColumnTransformer([("num", SimpleImputer(strategy="median"),
                                         num + ["nozzles_replaced"])]),
                     LogisticRegression(max_iter=1000))
f1 = cross_val_score(leak, df, y, cv=5, scoring="f1").mean()
print(f"{'with a leaking feature':28s} F1 = {f1:.3f}")
