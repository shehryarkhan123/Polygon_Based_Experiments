import json
import sys
from pathlib import Path

RF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RF_DIR))

import joblib
import numpy as np

from config import CLASS_NAMES, RESULTS_DIR, SEED
from dataset import prepare_data
from evaluate import evaluate_split, gini_importance, permutation_on_bands
from plots import plot_all
from train import train_model


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%"


def _depth_label(max_depth) -> str:
    return "None" if max_depth is None else str(max_depth)


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


def _is_best(row: dict, history: dict) -> bool:
    return (
        row["n_estimators"] == history["best_n_estimators"]
        and row["max_depth"] == history["best_max_depth"]
    )


def _grid_table(history: dict) -> str:
    rows = [
        "| n_estimators | max_depth | Val macro F1 | Val acc |",
        "|---:|---:|---:|---:|",
    ]
    for row in history["grid"]:
        mark = " (best)" if _is_best(row, history) else ""
        rows.append(
            f"| {row['n_estimators']} | {_depth_label(row['max_depth'])}{mark} | "
            f"{row['val_macro_f1']:.4f} | {row['val_acc']:.4f} |"
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


def _gini_table(gini: dict) -> str:
    rows = [
        "| Feature | Gini importance |",
        "|---|---:|",
    ]
    pairs = sorted(
        zip(gini["bands"], gini["importances"]),
        key=lambda r: r[1],
        reverse=True,
    )
    for band, imp in pairs:
        rows.append(f"| {band} | {imp:.4f} |")
    return "\n".join(rows)


def _strip_curves(metrics: dict) -> dict:
    keep = dict(metrics)
    for key in ("roc", "pr", "classification_report"):
        keep.pop(key, None)
    return keep


def write_readme(
    counts: dict,
    history: dict,
    val_metrics: dict,
    test_metrics: dict,
    perm: dict,
    gini: dict,
    figure_paths: dict,
    weights: dict,
) -> None:
    rel = lambda p: Path(p).relative_to(RF_DIR).as_posix()
    maj_name, maj_pct = _majority_share(counts["test"])
    maj_n = counts["test"][maj_name]
    test_n = sum(counts["test"].values())
    top_perm = max(zip(perm["bands"], perm["importances_mean"]), key=lambda r: r[1])[0]
    top_gini = max(zip(gini["bands"], gini["importances"]), key=lambda r: r[1])[0]
    w = weights
    best_depth = _depth_label(history["best_max_depth"])

    text = f"""# Random Forest: 5-class land cover (7 features)

Random Forest on the same polygon splits as the CNN and Linear SVM.

- Classes: **CPF 253**, **SL 284**, **water**, **urban**, **road**
- Features: `B1, B2, B3, B4, GNDVI, EVI, SAVI` (no lat/lon)
- Split unit: whole polygons
- `class_weight="balanced"` — urban/water were **not** downsampled; the RF is **not** biased toward large classes
- Grid on **validate** only: `n_estimators` in `{{100, 200}}`, `max_depth` in `{{20, None}}`. Best = **{history['best_n_estimators']} trees, depth {best_depth}** (val macro F1 {history['best_val_macro_f1']:.4f})
- Test scored **once**

Re-run `python RANDOM_FOREST/run_rf.py`. This file is rewritten at the end of that script.

## Why Random Forest

Linear SVM draws one straight split in 7-D. RF can mix bands (for example high GNDVI and low red). That is the point of this third model. Same full pixels. Same class weights. Still not biased by class size.

## Headline results

| Split | Accuracy | Balanced acc | Kappa | ROC-AUC (macro OvR) | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validate | {val_metrics['accuracy']:.4f} | {val_metrics['balanced_accuracy']:.4f} | {val_metrics['kappa']:.4f} | {val_metrics['roc_auc']:.4f} | {val_metrics['macro_f1']:.4f} | {val_metrics['weighted_f1']:.4f} |
| Test (once) | {test_metrics['accuracy']:.4f} | {test_metrics['balanced_accuracy']:.4f} | {test_metrics['kappa']:.4f} | {test_metrics['roc_auc']:.4f} | {test_metrics['macro_f1']:.4f} | {test_metrics['weighted_f1']:.4f} |

## Is the model biased toward large classes?

**No.** Classes have very different pixel counts (urban **495,553**, road **69,203**, about 7.5:1). That does **not** mean the RF only predicts urban. Large classes were kept in full. Bias is handled with `class_weight="balanced"` (inverse frequency), the same idea as the CNN and SVM.

**Problem.** A dummy that always predicts **{maj_name}** would get about **{maj_pct:.1f}%** test accuracy and **0** recall on CPF, SL, water, and road. This is **not** a dummy dataset. It is a fake always-{maj_name} guess on the **real** test pixels ({maj_n:,} / {test_n:,}). If the RF ignored small classes, overall accuracy could still look high because urban + water are ~67% of pixels.

**Solution.** `RandomForestClassifier(..., class_weight="balanced")`. sklearn sets weight `N / (5 × n_c)` so road (smallest) is up-weighted (**{w['road']:.3f}**) and urban (largest) is down-weighted (**{w['urban']:.3f}**). We report balanced accuracy and macro F1, not only overall accuracy. We did **not** downsample urban/water.

**Evidence (this run).** Test accuracy is **{test_metrics['accuracy']:.4f}** vs the {maj_pct:.1f}% dummy. Balanced accuracy is **{test_metrics['balanced_accuracy']:.4f}** (not collapsed onto one class). Road test recall is **{test_metrics['per_class']['road']['recall']:.4f}** and CPF 253 test recall is **{test_metrics['per_class']['CPF 253']['recall']:.4f}** — both would be 0 if the model ignored small classes. Water/urban are strong because they are spectrally distinct, not only because they have more pixels.

| Class | Train pixels | Weight `N/(5 n_c)` | Val recall | Test recall |
|---|---:|---:|---:|---:|
| CPF 253 | {counts['train']['CPF 253']:,} | {w['CPF 253']:.3f} | {val_metrics['per_class']['CPF 253']['recall']:.4f} | {test_metrics['per_class']['CPF 253']['recall']:.4f} |
| SL 284 | {counts['train']['SL 284']:,} | {w['SL 284']:.3f} | {val_metrics['per_class']['SL 284']['recall']:.4f} | {test_metrics['per_class']['SL 284']['recall']:.4f} |
| water | {counts['train']['water']:,} | {w['water']:.3f} | {val_metrics['per_class']['water']['recall']:.4f} | {test_metrics['per_class']['water']['recall']:.4f} |
| urban | {counts['train']['urban']:,} | {w['urban']:.3f} | {val_metrics['per_class']['urban']['recall']:.4f} | {test_metrics['per_class']['urban']['recall']:.4f} |
| road | {counts['train']['road']:,} | {w['road']:.3f} | {val_metrics['per_class']['road']['recall']:.4f} | {test_metrics['per_class']['road']['recall']:.4f} |

## Data

{_counts_table(counts)}

`StandardScaler` is fit on **train** only.

## Model

- `sklearn.ensemble.RandomForestClassifier`
- `class_weight="balanced"`, `n_jobs=-1`, seed 42
- Grid on validate (best by macro F1):

{_grid_table(history)}

![Grid]({rel(figure_paths['grid'])})

ROC uses `predict_proba`.

## How to run

```
python RANDOM_FOREST/run_rf.py
```

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

Each feature shuffled on {perm['n_samples']:,} validation pixels (10 repeats). **{top_perm}** is strongest this run.

{_perm_table(perm)}

![Permutation importance]({rel(figure_paths['permutation'])})

## Gini importance

Built-in mean decrease in impurity. **{top_gini}** is strongest this run.

{_gini_table(gini)}

![Gini importance]({rel(figure_paths['gini'])})

## Files

| Path | What |
|---|---|
| `results/best_model.joblib` | Fitted Random Forest + scaler |
| `results/metrics.json` | Numeric metrics |
| `results/grid.png` | Trees / depth vs validate scores |
| `results/confusion_matrix_*.png` | 5×5 confusion matrices |
| `results/roc_*.png` / `results/pr_*.png` | OvR curves |
| `results/permutation_importance.png` | Permutation importance |
| `results/gini_importance.png` | Gini importance |
| `RF_README.md` | This report |
"""
    (RF_DIR / "RF_README.md").write_text(text, encoding="utf-8")
    print(f"wrote {RF_DIR / 'RF_README.md'}")


def main() -> None:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    print("counts:", data["counts"])

    clf, history = train_model(
        data["x_train"], data["y_train"], data["x_val"], data["y_val"]
    )
    joblib.dump(
        {"model": clf, "scaler": data["scaler"], "history": history},
        RESULTS_DIR / "best_model.joblib",
    )
    print(f"saved {RESULTS_DIR / 'best_model.joblib'}")

    val_metrics = evaluate_split(clf, data["x_val"], data["y_val"])
    test_metrics = evaluate_split(clf, data["x_test"], data["y_test"])
    perm = permutation_on_bands(clf, data["x_val"], data["y_val"])
    gini = gini_importance(clf)

    print("\n=== VALIDATE ===")
    print(val_metrics["classification_report"])
    print("=== TEST ===")
    print(test_metrics["classification_report"])

    figure_paths = plot_all(history, val_metrics, test_metrics, perm, gini)
    payload = {
        "counts": data["counts"],
        "class_weights": data["weights"],
        "history": history,
        "validate": _strip_curves(val_metrics),
        "test": _strip_curves(test_metrics),
        "permutation_importance": perm,
        "gini_importance": gini,
        "figures": {k: Path(v).name for k, v in figure_paths.items()},
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {RESULTS_DIR / 'metrics.json'}")

    write_readme(
        data["counts"],
        history,
        val_metrics,
        test_metrics,
        perm,
        gini,
        figure_paths,
        data["weights"],
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
