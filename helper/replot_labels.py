"""Remap CPF/SL keys in saved metrics and redraw plots (no retraining)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RENAME = {"CPF": "CPF 253", "SL": "SL 284"}
CLASS_KEYS = {"CPF", "SL", "water", "urban", "road"}


def remap(obj):
    if isinstance(obj, dict):
        if CLASS_KEYS & obj.keys():
            return {RENAME.get(k, k): remap(v) for k, v in obj.items()}
        return {k: remap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap(x) for x in obj]
    return obj


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return remap(data)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"rewrote {path}")


def history_from(data: dict) -> dict:
    h = dict(data.get("history") or {})
    if "best_epoch" in data:
        h.setdefault("best_epoch", data["best_epoch"])
    return h


def import_plots(mod_dir: Path):
    for name in ("plots", "config"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(mod_dir))
    import plots as plots_mod

    sys.path.pop(0)
    return plots_mod


def replot_nn(mod_dir: Path) -> None:
    plots_mod = import_plots(mod_dir)
    data = load_json(mod_dir / "results" / "metrics.json")
    save_json(mod_dir / "results" / "metrics.json", data)
    hist = history_from(data)
    if "train_loss" in hist:
        plots_mod.plot_history(hist)
    plots_mod.plot_confusion(data["validate"], "validate")
    plots_mod.plot_confusion(data["test"], "test")
    if "permutation_importance" in data:
        plots_mod.plot_permutation(data["permutation_importance"])
    print(f"replotted {mod_dir.name}")


def main() -> None:
    replot_nn(ROOT / "CNN")
    replot_nn(ROOT / "TABULAR_TFRM")
    replot_nn(ROOT / "SPATIAL_TFRM")

    svm_plots = import_plots(ROOT / "SVM")
    data = load_json(ROOT / "SVM" / "results" / "metrics.json")
    save_json(ROOT / "SVM" / "results" / "metrics.json", data)
    svm_plots.plot_c_grid(data["history"])
    svm_plots.plot_confusion(data["validate"], "validate")
    svm_plots.plot_confusion(data["test"], "test")
    svm_plots.plot_permutation(data["permutation_importance"])
    print("replotted SVM")

    rf_plots = import_plots(ROOT / "RANDOM_FOREST")
    rf_path = ROOT / "RANDOM_FOREST" / "results" / "metrics.json"
    data = load_json(rf_path)
    save_json(rf_path, data)
    rf_plots.plot_grid(data["history"])
    rf_plots.plot_confusion(data["validate"], "validate")
    rf_plots.plot_confusion(data["test"], "test")
    rf_plots.plot_permutation(data["permutation_importance"])
    if "gini_importance" in data:
        rf_plots.plot_gini(data["gini_importance"])
    print("replotted RF")

    abl = ROOT / "SPATIAL_TFRM" / "ablation" / "results"
    if abl.exists():
        for path in abl.glob("*metrics.json"):
            save_json(path, load_json(path))
        summary = abl / "summary.json"
        if summary.exists():
            save_json(summary, load_json(summary))

    rename_markdown()


MD_FILES = [
    ROOT / "ReviewerComments.md",
    ROOT / "ReadME.md",
    ROOT / "CNN" / "CNN_README.md",
    ROOT / "CNN" / "CNN_README_2class.md",
    ROOT / "SVM" / "SVM_README.md",
    ROOT / "RANDOM_FOREST" / "RF_README.md",
    ROOT / "TABULAR_TFRM" / "TFR_README.md",
    ROOT / "SPATIAL_TFRM" / "SPATIAL_README.md",
    ROOT / "SPATIAL_TFRM" / "ablation" / "ABLATION_README.md",
]


def rename_markdown() -> None:
    for path in MD_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        updated = text.replace(
            "Training (train vs validate)", "Training set vs validation set"
        )
        updated = re.sub(r"\bCPF\b(?! 253)", "CPF 253", updated)
        updated = re.sub(r"\bSL\b(?! 284)", "SL 284", updated)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            print(f"updated {path}")


if __name__ == "__main__":
    main()
