# Decision tree on real medical data (Wisconsin Diagnostic Breast Cancer)
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score

data = load_breast_cancer()
X, y = data.data, (data.target == 0).astype(int)          # 1 = malignant tumour
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
for depth in (1, 2, 3, 6, None):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    p = clf.predict(X_te)
    print(f"depth {str(depth):4s} recall {recall_score(y_te, p):.3f} "
          f"precision {precision_score(y_te, p):.3f} leaves {clf.get_n_leaves()}")
best = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
print(export_text(best, feature_names=list(data.feature_names), decimals=3))
