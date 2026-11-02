import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

rows = [
    ("high","prolonged","high","normal","yes"), ("high","prolonged","normal","normal","yes"),
    ("high","normal","high","normal","yes"),
    ("high","normal","normal","high","no"),
    ("elevated","prolonged","high","normal","yes"),
    ("elevated","prolonged","normal","high","no"),
    ("elevated","normal","high","normal","yes"),
    ("elevated","normal","normal","normal","no"),
    ("normal","prolonged","normal","normal","no"), ("normal","prolonged","high","normal","yes"),
    ("normal","normal","normal","high","no"), ("normal","normal","normal","normal","no"),
    ("high","prolonged","high","high","yes"),
    ("normal","normal","high","high","no"),
]
df = pd.DataFrame(rows, columns=["egt", "start", "fuel", "vib", "fault"])
enc = {"normal": 0, "elevated": 1, "high": 2, "prolonged": 1}
X = df[["egt", "start", "fuel", "vib"]].replace(enc)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
clf.fit(X, df["fault"])
print(export_text(clf, feature_names=list(X.columns)))
