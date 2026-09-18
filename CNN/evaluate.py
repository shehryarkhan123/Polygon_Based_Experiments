import numpy as np
import torch
import torch.nn as nn
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
PRED_BATCH = 8192


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    probas = []
    x = np.asarray(x, dtype=np.float32)
    for i in range(0, len(x), PRED_BATCH):
        xb = torch.from_numpy(x[i : i + PRED_BATCH]).unsqueeze(1).to(device)
        logits = model(xb)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        preds.append(proba.argmax(axis=1))
        probas.append(proba)
    return np.concatenate(preds), np.concatenate(probas)


def split_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    roc_auc = float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=LABELS)
    )
    roc = {}
    pr = {}
    for i, name in enumerate(CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        prec_curve, rec_curve, _ = precision_recall_curve(y_bin, y_proba[:, i])
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


def permutation_on_bands(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> dict:
    rng = np.random.default_rng(SEED)
    n = min(PERM_SAMPLE_SIZE, len(y))
    idx = rng.choice(len(y), size=n, replace=False)
    xs, ys = x[idx].copy(), y[idx]
    baseline_pred, _ = predict(model, xs, device)
    baseline = accuracy_score(ys, baseline_pred)
    means = []
    stds = []
    for col in range(xs.shape[1]):
        drops = []
        for _ in range(PERM_REPEATS):
            shuffled = xs.copy()
            rng.shuffle(shuffled[:, col])
            pred, _ = predict(model, shuffled, device)
            drops.append(baseline - accuracy_score(ys, pred))
        means.append(float(np.mean(drops)))
        stds.append(float(np.std(drops)))
    return {
        "bands": BANDS,
        "importances_mean": means,
        "importances_std": stds,
        "n_samples": int(n),
    }


def evaluate_split(
    model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device
) -> dict:
    pred, proba = predict(model, x, device)
    return split_metrics(y, pred, proba)
