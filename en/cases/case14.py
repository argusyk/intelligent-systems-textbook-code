# Handwritten digit recognition with a multilayer perceptron
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = load_digits(return_X_y=True)                        # 1797 images of 8x8
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
for hidden in [(16,), (64,), (64, 32)]:
    net = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=hidden, early_stopping=True,
                                      max_iter=1000, random_state=0)).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, net.predict(X_te))
    print(f"hidden layers {str(hidden):9s} accuracy {acc:.3f}")
cm = confusion_matrix(y_te, net.predict(X_te))
off = [(i, j, int(cm[i, j])) for i in range(10) for j in range(10) if i != j and cm[i, j] > 1]
print("most frequent confusions (true, predicted, count):", off)
