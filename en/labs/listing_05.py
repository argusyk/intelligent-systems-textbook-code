from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

models = {
    "MLP (16)": make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(16,), early_stopping=True,
                                            max_iter=2000, random_state=0)),
    "tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=0),
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
}
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=5, scoring="f1_macro")
    print(f"{name:22s} {s.mean():.3f} +- {s.std():.3f}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
mlp = models["MLP (16)"].fit(X_tr, y_tr)
imp = permutation_importance(mlp, X_te, y_te, scoring="f1_macro", n_repeats=20, random_state=0)
print(imp.importances_mean.round(3))
