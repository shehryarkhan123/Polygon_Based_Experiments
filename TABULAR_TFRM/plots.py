import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from config import BANDS, CLASS_NAMES, RESULTS_DIR

SPLIT_LABEL = {"validate": "validation set", "test": "test set"}


def _save(fig: plt.Figure, name: str) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_history(history: dict) -> list[str]:
    epochs = range(1, len(history["train_loss"]) + 1)
    paths = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], label="training set")
    ax.plot(epochs, history["val_loss"], label="validation set")
    ax.set_title("FT-Transformer training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()
    paths.append(_save(fig, "loss_curve.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_acc"], label="training set")
    ax.plot(epochs, history["val_acc"], label="validation set")
    ax.set_title("FT-Transformer training and validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    paths.append(_save(fig, "accuracy_curve.png"))
    return paths


def plot_confusion(metrics: dict, split: str) -> str:
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"FT-Transformer confusion matrix ({SPLIT_LABEL.get(split, split)})")
    return _save(fig, f"confusion_matrix_{split}.png")


def plot_roc(metrics: dict, split: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in CLASS_NAMES:
        curve = metrics["roc"][name]
        ax.plot(curve["fpr"], curve["tpr"], label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"FT-Transformer ROC OvR ({SPLIT_LABEL.get(split, split)}), macro AUC = {metrics['roc_auc']:.3f}")
    ax.legend()
    return _save(fig, f"roc_{split}.png")


def plot_pr(metrics: dict, split: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in CLASS_NAMES:
        curve = metrics["pr"][name]
        ax.plot(curve["recall"], curve["precision"], label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"FT-Transformer precision-recall OvR ({SPLIT_LABEL.get(split, split)})")
    ax.legend()
    return _save(fig, f"pr_{split}.png")


def plot_permutation(perm: dict) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    means = np.array(perm["importances_mean"])
    stds = np.array(perm["importances_std"])
    ax.bar(BANDS, means, yerr=stds, capsize=4, color="#4C78A8")
    ax.set_title("FT-Transformer permutation importance (feature accuracy drop)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean accuracy decrease")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, "permutation_importance.png")


def plot_all(history: dict, val_metrics: dict, test_metrics: dict, perm: dict) -> dict:
    return {
        "history": plot_history(history),
        "confusion_validate": plot_confusion(val_metrics, "validate"),
        "confusion_test": plot_confusion(test_metrics, "test"),
        "roc_validate": plot_roc(val_metrics, "validate"),
        "roc_test": plot_roc(test_metrics, "test"),
        "pr_validate": plot_pr(val_metrics, "validate"),
        "pr_test": plot_pr(test_metrics, "test"),
        "permutation": plot_permutation(perm),
    }
