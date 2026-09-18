import numpy as np
from sklearn.metrics import f1_score

from config import MAX_DEPTH_GRID, N_CLASSES, N_ESTIMATORS_GRID
from model import make_rf


def _depth_label(max_depth) -> str:
    return "None" if max_depth is None else str(max_depth)


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    labels = list(range(N_CLASSES))
    history = []
    best_clf = None
    best_n = None
    best_depth = None
    best_f1 = -1.0

    for n_estimators in N_ESTIMATORS_GRID:
        for max_depth in MAX_DEPTH_GRID:
            print(f"fitting RF n_estimators={n_estimators} max_depth={_depth_label(max_depth)} ...")
            clf = make_rf(n_estimators, max_depth)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_val)
            f1 = float(f1_score(y_val, pred, average="macro", labels=labels, zero_division=0))
            acc = float((pred == y_val).mean())
            row = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "val_macro_f1": f1,
                "val_acc": acc,
            }
            history.append(row)
            print(
                f"  n={n_estimators} depth={_depth_label(max_depth)}  "
                f"val macro F1 {f1:.4f}  val acc {acc:.4f}"
            )
            if f1 > best_f1:
                best_f1 = f1
                best_n = n_estimators
                best_depth = max_depth
                best_clf = clf

    print(
        f"best n_estimators={best_n} max_depth={_depth_label(best_depth)} "
        f"(val macro F1 {best_f1:.4f})"
    )
    return best_clf, {
        "best_n_estimators": best_n,
        "best_max_depth": best_depth,
        "best_val_macro_f1": best_f1,
        "grid": history,
    }
