import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from config import BANDS, CLASS_NAMES, N_CLASSES, PERM_REPEATS, PERM_SAMPLE_SIZE, SEED

LABELS = list(range(N_CLASSES))


def scores(clf, x: np.ndarray) -> np.ndarray:
    return np.asarray(clf.predict_proba(x), dtype=np.float64)


def split_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    roc_auc = float(
        roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=LABELS)
    )
    roc = {}
    pr = {}
    for i, name in enumerate(CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_score[:, i])
        prec_curve, rec_curve, _ = precision_recall_curve(y_bin, y_score[:, i])
        roc[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        pr[name] = {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0)
        ),
        "classification_report": report,
        "roc": roc,
        "pr": pr,
    }


def permutation_on_bands(clf, x: np.ndarray, y: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    n = min(PERM_SAMPLE_SIZE, len(y))
    idx = rng.choice(len(y), size=n, replace=False)
    xs, ys = x[idx].copy(), y[idx]
    baseline = accuracy_score(ys, clf.predict(xs))
    means = []
    stds = []
    for col in range(xs.shape[1]):
        drops = []
        for _ in range(PERM_REPEATS):
            shuffled = xs.copy()
            rng.shuffle(shuffled[:, col])
            drops.append(baseline - accuracy_score(ys, clf.predict(shuffled)))
        means.append(float(np.mean(drops)))
        stds.append(float(np.std(drops)))
    return {
        "bands": BANDS,
        "importances_mean": means,
        "importances_std": stds,
        "n_samples": int(n),
    }


def gini_importance(clf) -> dict:
    return {
        "bands": BANDS,
        "importances": [float(v) for v in clf.feature_importances_],
    }


def evaluate_split(clf, x: np.ndarray, y: np.ndarray) -> dict:
    pred = clf.predict(x)
    y_score = scores(clf, x)
    return split_metrics(y, pred, y_score)
