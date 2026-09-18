import json
import random
import sys
from pathlib import Path

ABLATION_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = ABLATION_DIR.parent
sys.path.insert(0, str(SPATIAL_DIR))

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config import (
    ABLATION_RESULTS_DIR,
    BANDS,
    CLASS_NAMES,
    N_CLASSES,
    N_FEATURES,
    RESULTS_DIR,
    RF_MAX_DEPTH_GRID,
    RF_N_ESTIMATORS_GRID,
    SEED,
)
from dataset import (
    center_tokens,
    flatten_patches,
    make_loader,
    patches_to_chw,
    prepare_data,
)
from evaluate import evaluate_split, split_metrics
from model import FlattenMLP, PatchCNN2D, SpatialContextTransformer
from train import train_model

LABELS = list(range(N_CLASSES))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _strip_curves(metrics: dict) -> dict:
    keep = dict(metrics)
    for key in ("roc", "pr", "y_true", "y_pred", "y_score", "classification_report"):
        keep.pop(key, None)
    return keep


def _n_test(metrics: dict) -> int:
    return sum(row["support"] for row in metrics["per_class"].values())


def row_from_test(run_id: str, name: str, test_metrics: dict, note: str = "") -> dict:
    pc = test_metrics["per_class"]
    return {
        "id": run_id,
        "name": name,
        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "cpf_f1": pc["CPF 253"]["f1"],
        "sl_f1": pc["SL 284"]["f1"],
        "n_test": _n_test(test_metrics),
        "note": note,
    }


def save_run(run_id: str, payload: dict) -> None:
    ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ABLATION_RESULTS_DIR / f"{run_id}_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {path}")


def load_main_spatial_b() -> dict | None:
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("n_features") != N_FEATURES:
        return None
    if data.get("bands") != list(BANDS):
        return None
    return data


def train_torch(
    run_id: str,
    name: str,
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    weights: torch.Tensor,
    device: torch.device,
    note: str,
) -> dict:
    print(f"\n=== {run_id}: {name} ===")
    train_loader = make_loader(x_train, y_train, shuffle=True)
    val_loader = make_loader(x_val, y_val, shuffle=False)
    model, history = train_model(
        train_loader, val_loader, weights, device, model=model
    )
    ckpt = ABLATION_RESULTS_DIR / f"{run_id}_best_model.pt"
    ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history}, ckpt)
    val_metrics = evaluate_split(model, x_val, y_val, device)
    test_metrics = evaluate_split(model, x_test, y_test, device)
    print(test_metrics["classification_report"])
    payload = {
        "id": run_id,
        "name": name,
        "note": note,
        "bands": list(BANDS),
        "n_features": N_FEATURES,
        "history": {k: v for k, v in history.items() if k != "best_epoch"},
        "best_epoch": history["best_epoch"],
        "validate": _strip_curves(val_metrics),
        "test": _strip_curves(test_metrics),
    }
    save_run(run_id, payload)
    return row_from_test(run_id, name, test_metrics, note)


def _depth_label(max_depth) -> str:
    return "None" if max_depth is None else str(max_depth)


def train_rf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    print("\n=== E: 3x3 Random Forest ===")
    x_train_f = flatten_patches(x_train)
    x_val_f = flatten_patches(x_val)
    x_test_f = flatten_patches(x_test)
    history = []
    best_clf = None
    best_n = None
    best_depth = None
    best_f1 = -1.0
    for n_estimators in RF_N_ESTIMATORS_GRID:
        for max_depth in RF_MAX_DEPTH_GRID:
            print(
                f"fitting RF n_estimators={n_estimators} "
                f"max_depth={_depth_label(max_depth)} ..."
            )
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEED,
            )
            clf.fit(x_train_f, y_train)
            pred = clf.predict(x_val_f)
            f1 = float(
                f1_score(y_val, pred, average="macro", labels=LABELS, zero_division=0)
            )
            acc = float((pred == y_val).mean())
            history.append(
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "val_macro_f1": f1,
                    "val_acc": acc,
                }
            )
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
    val_pred = best_clf.predict(x_val_f)
    val_proba = best_clf.predict_proba(x_val_f)
    test_pred = best_clf.predict(x_test_f)
    test_proba = best_clf.predict_proba(x_test_f)
    val_metrics = split_metrics(y_val, val_pred, val_proba)
    test_metrics = split_metrics(y_test, test_pred, test_proba)
    print(test_metrics["classification_report"])
    note = (
        f"3x3 RF on flattened {x_train_f.shape[1]} features; "
        f"trees={best_n}, depth={_depth_label(best_depth)} by val macro F1"
    )
    payload = {
        "id": "E",
        "name": "3x3 RF",
        "note": note,
        "bands": list(BANDS),
        "n_features": N_FEATURES,
        "best_n_estimators": best_n,
        "best_max_depth": best_depth,
        "grid": history,
        "validate": _strip_curves(val_metrics),
        "test": _strip_curves(test_metrics),
    }
    save_run("E", payload)
    return row_from_test("E", "3x3 RF", test_metrics, note)


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def write_readme(rows: list[dict], device: str) -> None:
    same = [r for r in rows if r["id"] in {"A", "B", "C", "D", "E"}]
    windows = [r for r in rows if r["id"] in {"A", "B", "W5"}]
    lines = [
        "# Spatial ablation: Transformer vs spatial context (B1–B4)",
        "",
        "Same polygon split as the spatial transformer. Features are **B1, B2, B3, B4** only.",
        "A–E use the **same 3×3-kept pixels**. The 5×5 Transformer uses its own kept set (more edge drop).",
        "",
        f"Device: `{device}`. Test scored **once**. Settings on validate only.",
        "",
        "Re-run `python SPATIAL_TFRM/ablation/run_ablation.py`.",
        "",
        "## Same 3×3 pixels",
        "",
        "| ID | Model | Test acc | Balanced acc | Macro F1 | CPF 253 F1 | SL 284 F1 | n test |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in same:
        lines.append(
            f"| {r['id']} | {r['name']} | {_fmt(r['accuracy'])} | "
            f"{_fmt(r['balanced_accuracy'])} | {_fmt(r['macro_f1'])} | "
            f"{_fmt(r['cpf_f1'])} | {_fmt(r['sl_f1'])} | {r['n_test']:,} |"
        )
    lines.extend(
        [
            "",
            "A vs B: neighbourhood (same Transformer). B vs C/D/E: architecture (same 3×3).",
            "",
            "## Window size",
            "",
            "| ID | Model | Test acc | Balanced acc | Macro F1 | CPF 253 F1 | SL 284 F1 | n test |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in windows:
        lines.append(
            f"| {r['id']} | {r['name']} | {_fmt(r['accuracy'])} | "
            f"{_fmt(r['balanced_accuracy'])} | {_fmt(r['macro_f1'])} | "
            f"{_fmt(r['cpf_f1'])} | {_fmt(r['sl_f1'])} | {r['n_test']:,} |"
        )
    lines.extend(
        [
            "",
            "A is centre-only on the 3×3-kept pixels (not a separate 1×1 load, which would keep field edges).",
            "W5 drops extra edge pixels, especially road, so n test is smaller.",
            "",
            "## Runs",
            "",
            "| ID | What | Input |",
            "|---|---|---|",
            "| A | Same Transformer, centre pixel | `(N, 1, 4)` |",
            "| B | Proposed 3×3 Transformer | `(N, 9, 4)` |",
            "| C | MLP | flatten 36 |",
            "| D | 2D CNN | `(N, 4, 3, 3)` |",
            "| E | Random Forest | flatten 36 |",
            "| W5 | Same Transformer, 5×5 | `(N, 25, 4)` |",
            "",
        ]
    )
    path = ABLATION_DIR / "ABLATION_README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    set_seed(SEED)
    ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"device: {device}")

    data = prepare_data(radius=1)
    weights = data["weights"]
    x_tr, y_tr = data["x_train"], data["y_train"]
    x_va, y_va = data["x_val"], data["y_val"]
    x_te, y_te = data["x_test"], data["y_test"]
    rows: list[dict] = []

    x_tr_c = center_tokens(x_tr)
    x_va_c = center_tokens(x_va)
    x_te_c = center_tokens(x_te)
    rows.append(
        train_torch(
            "A",
            "1x1 Transformer (centre of 3x3)",
            SpatialContextTransformer(),
            x_tr_c,
            y_tr,
            x_va_c,
            y_va,
            x_te_c,
            y_te,
            weights,
            device,
            "Same net as B; centre token only from the 3x3 patches",
        )
    )

    main_b = load_main_spatial_b()
    if main_b is not None:
        print("\n=== B: 3x3 Transformer (from SPATIAL_TFRM/results) ===")
        rows.append(
            row_from_test(
                "B",
                "3x3 Transformer",
                main_b["test"],
                "Proposed spatial method; reused 4-band run_spatial.py",
            )
        )
        save_run(
            "B",
            {
                "id": "B",
                "name": "3x3 Transformer",
                "note": "reused SPATIAL_TFRM/results/metrics.json",
                "source": str(RESULTS_DIR / "metrics.json"),
                "bands": list(BANDS),
                "n_features": N_FEATURES,
                "validate": main_b["validate"],
                "test": main_b["test"],
            },
        )
    else:
        rows.append(
            train_torch(
                "B",
                "3x3 Transformer",
                SpatialContextTransformer(),
                x_tr,
                y_tr,
                x_va,
                y_va,
                x_te,
                y_te,
                weights,
                device,
                "Proposed spatial method",
            )
        )

    rows.append(
        train_torch(
            "C",
            "3x3 MLP",
            FlattenMLP(),
            x_tr,
            y_tr,
            x_va,
            y_va,
            x_te,
            y_te,
            weights,
            device,
            "Flatten 9x4 then 36 to 64 to 5",
        )
    )

    x_tr_cnn = patches_to_chw(x_tr)
    x_va_cnn = patches_to_chw(x_va)
    x_te_cnn = patches_to_chw(x_te)
    rows.append(
        train_torch(
            "D",
            "3x3 CNN",
            PatchCNN2D(),
            x_tr_cnn,
            y_tr,
            x_va_cnn,
            y_va,
            x_te_cnn,
            y_te,
            weights,
            device,
            "2D CNN on (4, 3, 3); not the pixel 1D CNN",
        )
    )

    rows.append(train_rf(x_tr, y_tr, x_va, y_va, x_te, y_te))

    data5 = prepare_data(radius=2)
    rows.append(
        train_torch(
            "W5",
            "5x5 Transformer",
            SpatialContextTransformer(),
            data5["x_train"],
            data5["y_train"],
            data5["x_val"],
            data5["y_val"],
            data5["x_test"],
            data5["y_test"],
            data5["weights"],
            device,
            "Same Transformer; 25 tokens; own 5x5-kept pixels",
        )
    )

    summary_path = ABLATION_RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {summary_path}")
    write_readme(rows, str(device))
    print("\nDone.")


if __name__ == "__main__":
    main()
