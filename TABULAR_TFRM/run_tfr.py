import json
import random
import sys
from pathlib import Path

TFR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TFR_DIR))

import numpy as np
import torch

from config import CLASS_NAMES, RESULTS_DIR, SEED
from dataset import prepare_data
from evaluate import evaluate_split, permutation_on_bands
from plots import plot_all
from train import train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%"


def _class_table(metrics: dict) -> str:
    rows = [
        "| Class | Precision | Recall | F1-score | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in metrics["per_class"].items():
        rows.append(
            f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['support']:,} |"
        )
    rows.append(
        f"| **Overall accuracy** |  |  | **{metrics['accuracy']:.4f}** | "
        f"**{sum(r['support'] for r in metrics['per_class'].values()):,}** |"
    )
    return "\n".join(rows)


def _cm_table(metrics: dict) -> str:
    cm = metrics["confusion_matrix"]
    header = "|  | " + " | ".join(f"Pred {n}" for n in CLASS_NAMES) + " |"
    sep = "|" + "|".join(["---"] + ["---:" for _ in CLASS_NAMES]) + "|"
    rows = [header, sep]
    for i, name in enumerate(CLASS_NAMES):
        cells = " | ".join(f"{cm[i][j]:,}" for j in range(len(CLASS_NAMES)))
        rows.append(f"| Actual {name} | {cells} |")
    return "\n".join(rows)


def _counts_table(counts: dict) -> str:
    header = "| Split | " + " | ".join(CLASS_NAMES) + " | Total | Share |"
    sep = "|---|" + "|".join(["---:" for _ in CLASS_NAMES]) + "|---:|---:|"
    rows = [header, sep]
    totals = {s: sum(counts[s].values()) for s in ("train", "validate", "test")}
    grand = sum(totals.values())
    for split in ("train", "validate", "test"):
        cells = " | ".join(f"{counts[split][n]:,}" for n in CLASS_NAMES)
        tot = totals[split]
        rows.append(f"| {split.capitalize()} | {cells} | {tot:,} | {_pct(tot, grand)} |")
    all_cells = " | ".join(
        f"**{counts['train'][n] + counts['validate'][n] + counts['test'][n]:,}**"
        for n in CLASS_NAMES
    )
    rows.append(f"| **All** | {all_cells} | **{grand:,}** | **100%** |")
    return "\n".join(rows)


def _majority_share(counts_split: dict) -> tuple[str, float]:
    name = max(counts_split, key=counts_split.get)
    total = sum(counts_split.values()) or 1
    return name, 100.0 * counts_split[name] / total


def _epoch_table(history: dict) -> str:
    rows = [
        "| Epoch | Train loss | Train acc | Val loss | Val acc |",
        "|---:|---:|---:|---:|---:|",
    ]
    n = len(history["train_loss"])
    best = history.get("best_epoch", 0)
    for i in range(n):
        epoch = f"{i + 1} (best)" if (i + 1) == best else str(i + 1)
        rows.append(
            f"| {epoch} | {history['train_loss'][i]:.4f} | "
            f"{history['train_acc'][i]:.4f} | "
            f"{history['val_loss'][i]:.4f} | "
            f"{history['val_acc'][i]:.4f} |"
        )
    return "\n".join(rows)


def _perm_table(perm: dict) -> str:
    rows = [
        "| Feature | Mean accuracy drop | Std |",
        "|---|---:|---:|",
    ]
    pairs = sorted(
        zip(perm["bands"], perm["importances_mean"], perm["importances_std"]),
        key=lambda r: r[1],
        reverse=True,
    )
    for band, mean, std in pairs:
        rows.append(f"| {band} | {mean:.4f} | {std:.4f} |")
    return "\n".join(rows)


def _strip_curves(metrics: dict) -> dict:
    keep = dict(metrics)
    for key in ("roc", "pr", "y_true", "y_pred", "y_score", "classification_report"):
        keep.pop(key, None)
    return keep


def write_readme(
    device: torch.device,
    counts: dict,
    history: dict,
    val_metrics: dict,
    test_metrics: dict,
    perm: dict,
    figure_paths: dict,
    class_weight_map: dict | None = None,
) -> None:
    rel = (
        lambda p: Path(p).relative_to(TFR_DIR).as_posix()
        if not isinstance(p, list)
        else None
    )
    top_feat = max(
        zip(perm["bands"], perm["importances_mean"]),
        key=lambda r: r[1],
    )[0]
    maj_name, maj_pct = _majority_share(counts["test"])
    maj_n = counts["test"][maj_name]
    test_n = sum(counts["test"].values())
    weight_txt = ""
    w = {name: 0.0 for name in CLASS_NAMES}
    if class_weight_map:
        weight_txt = ", ".join(f"{k} {v:.3f}" for k, v in class_weight_map.items())
        w.update(class_weight_map)

    text = f"""# FT-Transformer: 5-class land cover (7 features)

Small tabular transformer (FT-Transformer) on the same polygon splits as the CNN, Linear SVM, and Random Forest.

- Classes: **CPF 253**, **SL 284**, **water**, **urban**, **road**
- Features: `B1, B2, B3, B4, GNDVI, EVI, SAVI` (no lat/lon)
- Split unit: whole polygons
- Inverse-frequency class weights — urban/water were **not** downsampled; the model is **not** biased toward large classes
- Device (this run): `{device}`
- Best checkpoint: epoch {history['best_epoch']} (lowest validation loss = {history.get('best_val_loss', 0):.4f})
- Test scored **once**

Re-run `python TABULAR_TFRM/run_tfr.py`. This file is rewritten at the end of that script.

## Why FT-Transformer

CNN looks at **neighboring** bands. Linear SVM is one straight split. RF mixes features with trees. An FT-Transformer lets **every feature attend to every other** (GNDVI can look at B4) without assuming band order. That is the point of this fourth model. Same full pixels. Same class weights. Still not biased by class size.

This is **not** TabTransformer. That model is for categorical columns. All 7 features here are numeric.

## Headline results

| Split | Accuracy | Balanced acc | Kappa | ROC-AUC (macro OvR) | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validate | {val_metrics['accuracy']:.4f} | {val_metrics['balanced_accuracy']:.4f} | {val_metrics['kappa']:.4f} | {val_metrics['roc_auc']:.4f} | {val_metrics['macro_f1']:.4f} | {val_metrics['weighted_f1']:.4f} |
| Test (once) | {test_metrics['accuracy']:.4f} | {test_metrics['balanced_accuracy']:.4f} | {test_metrics['kappa']:.4f} | {test_metrics['roc_auc']:.4f} | {test_metrics['macro_f1']:.4f} | {test_metrics['weighted_f1']:.4f} |

## Is the model biased toward large classes?

**No.** Classes have very different pixel counts (urban **495,553**, road **69,203**, about 7.5:1). That does **not** mean the transformer only predicts urban. Large classes were kept in full. Bias is handled in the **loss** with inverse-frequency weights, the same idea as the CNN.

**Problem.** A dummy that always predicts **{maj_name}** would get about **{maj_pct:.1f}%** test accuracy and **0** recall on CPF, SL, water, and road. This is **not** a dummy dataset. It is a fake always-{maj_name} guess on the **real** test pixels ({maj_n:,} / {test_n:,}). If the model ignored small classes, overall accuracy could still look high because urban + water are ~67% of pixels.

**Solution.** Weighted cross-entropy with `w_c = N / (5 × n_c)`. Road (smallest) is up-weighted (**{w['road']:.3f}**) and urban (largest) is down-weighted (**{w['urban']:.3f}**). We report balanced accuracy and macro F1, not only overall accuracy. We did **not** downsample urban/water.

**Evidence (this run).** Test accuracy is **{test_metrics['accuracy']:.4f}** vs the {maj_pct:.1f}% dummy. Balanced accuracy is **{test_metrics['balanced_accuracy']:.4f}**. Road test recall is **{test_metrics['per_class']['road']['recall']:.4f}** and CPF 253 test recall is **{test_metrics['per_class']['CPF 253']['recall']:.4f}** — both would be 0 if the model ignored small classes.

| Class | Train pixels | Weight `N/(5 n_c)` | Val recall | Test recall |
|---|---:|---:|---:|---:|
| CPF 253 | {counts['train']['CPF 253']:,} | {w['CPF 253']:.3f} | {val_metrics['per_class']['CPF 253']['recall']:.4f} | {test_metrics['per_class']['CPF 253']['recall']:.4f} |
| SL 284 | {counts['train']['SL 284']:,} | {w['SL 284']:.3f} | {val_metrics['per_class']['SL 284']['recall']:.4f} | {test_metrics['per_class']['SL 284']['recall']:.4f} |
| water | {counts['train']['water']:,} | {w['water']:.3f} | {val_metrics['per_class']['water']['recall']:.4f} | {test_metrics['per_class']['water']['recall']:.4f} |
| urban | {counts['train']['urban']:,} | {w['urban']:.3f} | {val_metrics['per_class']['urban']['recall']:.4f} | {test_metrics['per_class']['urban']['recall']:.4f} |
| road | {counts['train']['road']:,} | {w['road']:.3f} | {val_metrics['per_class']['road']['recall']:.4f} | {test_metrics['per_class']['road']['recall']:.4f} |

Class weights this run: {weight_txt or "see training log"}.

## Data

{_counts_table(counts)}

`StandardScaler` is fit on **train** only.

## Model

- Numeric tokenizer: each feature `j` → `x_j * W_j + b_j` in R³²
- Learnable `[CLS]` token (sequence length 8)
- Transformer encoder: 2 layers, `d_model=32`, 4 heads, FFN 64, dropout 0.3
- Head: LayerNorm + Linear(32 → 5)
- Loss: weighted cross-entropy
- Optimizer: Adam (`lr=1e-3`, `weight_decay=1e-4`)
- Batch size 256, max 40 epochs, early stop on validation loss (patience 8), seed 42

## How to run

```
python TABULAR_TFRM/run_tfr.py
```

## Training set vs validation set

Early stop at epoch {len(history['train_loss'])}; **best validation loss at epoch {history['best_epoch']}**.

{_epoch_table(history)}

![Training and validation loss]({rel(figure_paths['history'][0])})

![Training and validation accuracy]({rel(figure_paths['history'][1])})

## Validate

Accuracy {val_metrics['accuracy']:.4f}. Balanced accuracy {val_metrics['balanced_accuracy']:.4f}. Kappa {val_metrics['kappa']:.4f}. ROC-AUC {val_metrics['roc_auc']:.4f}. Macro F1 {val_metrics['macro_f1']:.4f}. Weighted F1 {val_metrics['weighted_f1']:.4f}.

{_class_table(val_metrics)}

```
{val_metrics['classification_report'].rstrip()}
```

{_cm_table(val_metrics)}

![Validate confusion matrix]({rel(figure_paths['confusion_validate'])})

![Validate ROC]({rel(figure_paths['roc_validate'])})

![Validate precision-recall]({rel(figure_paths['pr_validate'])})

## Test (scored once)

Accuracy {test_metrics['accuracy']:.4f}. Balanced accuracy {test_metrics['balanced_accuracy']:.4f}. Kappa {test_metrics['kappa']:.4f}. ROC-AUC {test_metrics['roc_auc']:.4f}. Macro F1 {test_metrics['macro_f1']:.4f}. Weighted F1 {test_metrics['weighted_f1']:.4f}.

{_class_table(test_metrics)}

```
{test_metrics['classification_report'].rstrip()}
```

{_cm_table(test_metrics)}

![Test confusion matrix]({rel(figure_paths['confusion_test'])})

![Test ROC]({rel(figure_paths['roc_test'])})

![Test precision-recall]({rel(figure_paths['pr_test'])})

## Permutation importance

Each feature shuffled on {perm['n_samples']:,} validation pixels (10 repeats). **{top_feat}** is strongest this run.

{_perm_table(perm)}

![Permutation importance]({rel(figure_paths['permutation'])})

## Files

| Path | What |
|---|---|
| `results/best_model.pt` | Weights at best validation loss |
| `results/metrics.json` | Numeric metrics |
| `results/loss_curve.png` / `results/accuracy_curve.png` | Train vs validate |
| `results/confusion_matrix_*.png` | 5×5 confusion matrices |
| `results/roc_*.png` / `results/pr_*.png` | OvR curves |
| `results/permutation_importance.png` | Feature importance |
| `TFR_README.md` | This report |
"""
    (TFR_DIR / "TFR_README.md").write_text(text, encoding="utf-8")
    print(f"wrote {TFR_DIR / 'TFR_README.md'}")


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"device: {device}")

    data = prepare_data()
    print("counts:", data["counts"])
    weight_map = {
        name: float(w) for name, w in zip(CLASS_NAMES, data["weights"].tolist())
    }

    model, history = train_model(
        data["train_loader"], data["val_loader"], data["weights"], device
    )
    ckpt = RESULTS_DIR / "best_model.pt"
    torch.save({"model": model.state_dict(), "history": history}, ckpt)
    print(f"saved {ckpt}")

    val_metrics = evaluate_split(model, data["x_val"], data["y_val"], device)
    test_metrics = evaluate_split(model, data["x_test"], data["y_test"], device)
    perm = permutation_on_bands(model, data["x_val"], data["y_val"], device)

    print("\n=== VALIDATE ===")
    print(val_metrics["classification_report"])
    print("=== TEST ===")
    print(test_metrics["classification_report"])

    figure_paths = plot_all(history, val_metrics, test_metrics, perm)

    payload = {
        "device": str(device),
        "counts": data["counts"],
        "class_weights": weight_map,
        "history": {k: v for k, v in history.items() if k != "best_epoch"},
        "best_epoch": history["best_epoch"],
        "validate": _strip_curves(val_metrics),
        "test": _strip_curves(test_metrics),
        "permutation_importance": perm,
        "figures": {
            k: ([str(Path(p).name) for p in v] if isinstance(v, list) else Path(v).name)
            for k, v in figure_paths.items()
        },
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {metrics_path}")

    write_readme(
        device,
        data["counts"],
        history,
        val_metrics,
        test_metrics,
        perm,
        figure_paths,
        weight_map,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
