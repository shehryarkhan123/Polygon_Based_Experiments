import numpy as np
from sklearn.metrics import f1_score

from config import C_GRID, N_CLASSES
from model import make_svm


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    labels = list(range(N_CLASSES))
    history = []
    best_clf = None
    best_c = None
    best_f1 = -1.0

    for C in C_GRID:
        print(f"fitting LinearSVC C={C} ...")
        clf = make_svm(C)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_val)
        f1 = float(f1_score(y_val, pred, average="macro", labels=labels, zero_division=0))
        acc = float((pred == y_val).mean())
        n_iter = clf.n_iter_
        n_iter = int(np.max(n_iter)) if np.ndim(n_iter) else int(n_iter)
        row = {"C": C, "val_macro_f1": f1, "val_acc": acc, "n_iter": n_iter}
        history.append(row)
        print(f"  C={C}  val macro F1 {f1:.4f}  val acc {acc:.4f}  n_iter {clf.n_iter_}")
        if f1 > best_f1:
            best_f1 = f1
            best_c = C
            best_clf = clf

    print(f"best C={best_c} (val macro F1 {best_f1:.4f})")
    return best_clf, {"best_C": best_c, "best_val_macro_f1": best_f1, "grid": history}
