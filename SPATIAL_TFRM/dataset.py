from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BANDS,
    BATCH_SIZE,
    CLASS_DIRS,
    CLASS_NAMES,
    N_CLASSES,
    N_FEATURES,
    WINDOW_RADIUS,
    XY_COLS,
    center_index_for_radius,
    n_neighbors_for_radius,
    shifts_for_radius,
)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    missing = [c for c in (*XY_COLS, *BANDS) if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing columns {missing}")
    return df


def neighborhood_patches(df: pd.DataFrame, radius: int = WINDOW_RADIUS) -> np.ndarray:
    """Patches inside one polygon. Drops edge pixels without a full window."""
    shifts = shifts_for_radius(radius)
    n_neighbors = n_neighbors_for_radius(radius)
    part = df[list(XY_COLS) + list(BANDS)].copy()
    part["File X"] = part["File X"].round().astype(np.int64)
    part["File Y"] = part["File Y"].round().astype(np.int64)
    indexed = part.set_index(XY_COLS).sort_index()
    indexed = indexed[~indexed.index.duplicated(keep="first")]

    aligned = None
    for dx, dy in shifts:
        shifted = indexed.copy()
        shifted.index = pd.MultiIndex.from_tuples(
            [(x - dx, y - dy) for x, y in shifted.index],
            names=XY_COLS,
        )
        piece = shifted[list(BANDS)].copy()
        piece.columns = [f"{b}_{dx}_{dy}" for b in BANDS]
        aligned = piece if aligned is None else aligned.join(piece, how="inner")

    if aligned is None or aligned.empty:
        return np.zeros((0, n_neighbors, N_FEATURES), dtype=np.float32)
    arr = aligned.to_numpy(dtype=np.float32)
    return arr.reshape(len(aligned), n_neighbors, N_FEATURES)


def _patches_from_polygon_dir(
    class_dir: Path, split: str, radius: int = WINDOW_RADIUS
) -> tuple[np.ndarray, int]:
    folder = class_dir / split
    files = sorted(folder.glob("Polygon_*.csv"))
    if not files:
        raise FileNotFoundError(f"No polygon CSVs in {folder}")
    patches = []
    raw = 0
    n_neighbors = n_neighbors_for_radius(radius)
    for path in files:
        df = _read_csv(path)
        raw += len(df)
        patch = neighborhood_patches(df, radius=radius)
        if len(patch):
            patches.append(patch)
    if not patches:
        return np.zeros((0, n_neighbors, N_FEATURES), dtype=np.float32), raw
    return np.concatenate(patches, axis=0), raw


def load_split(
    split: str, radius: int = WINDOW_RADIUS
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    side = 2 * radius + 1
    xs = []
    ys = []
    kept = {}
    raw = {}
    for i, name in enumerate(CLASS_NAMES):
        patches, n_raw = _patches_from_polygon_dir(
            CLASS_DIRS[name], split, radius=radius
        )
        xs.append(patches)
        ys.append(np.full(len(patches), i, dtype=np.int64))
        kept[name] = int(len(patches))
        raw[name] = int(n_raw)
        print(
            f"  {split}/{name}: {n_raw:,} pixels -> {len(patches):,} with full {side}x{side}"
        )
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return x, y, kept, raw


def class_counts(y: np.ndarray) -> dict[str, int]:
    return {name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES)}


def class_weights(y_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
    weights = counts.sum() / (N_CLASSES * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    tensor_x = torch.tensor(np.array(x, copy=True), dtype=torch.float32)
    tensor_y = torch.tensor(np.array(y, copy=True), dtype=torch.int64)
    return DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def _scale_patches(
    x: np.ndarray, scaler: StandardScaler, fit: bool, radius: int
) -> np.ndarray:
    n = len(x)
    n_neighbors = n_neighbors_for_radius(radius)
    center = center_index_for_radius(radius)
    if fit:
        scaler.fit(x[:, center, :])
    flat = scaler.transform(x.reshape(-1, N_FEATURES)).astype(np.float32)
    return flat.reshape(n, n_neighbors, N_FEATURES)


def center_tokens(x: np.ndarray, radius: int = WINDOW_RADIUS) -> np.ndarray:
    """Keep the centre spectrum only: (N, 1, F) from the same patches."""
    c = center_index_for_radius(radius)
    return x[:, c : c + 1, :].copy()


def flatten_patches(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1).copy()


def patches_to_chw(x: np.ndarray) -> np.ndarray:
    """(N, side*side, C) -> (N, C, side, side) for a 2D CNN."""
    n, tokens, feats = x.shape
    side = int(round(tokens**0.5))
    if side * side != tokens:
        raise ValueError(f"token count {tokens} is not a square window")
    return np.transpose(x.reshape(n, side, side, feats), (0, 3, 1, 2)).copy()


def prepare_data(radius: int = WINDOW_RADIUS) -> dict:
    side = 2 * radius + 1
    print(f"building {side}x{side} patches inside each polygon ...")
    x_train, y_train, kept_train, raw_train = load_split("train", radius=radius)
    x_val, y_val, kept_val, raw_val = load_split("validate", radius=radius)
    x_test, y_test, kept_test, raw_test = load_split("test", radius=radius)

    scaler = StandardScaler()
    x_train = _scale_patches(x_train, scaler, fit=True, radius=radius)
    x_val = _scale_patches(x_val, scaler, fit=False, radius=radius)
    x_test = _scale_patches(x_test, scaler, fit=False, radius=radius)

    weights = class_weights(y_train)
    print(
        f"class weights (after {side}x{side} filter):",
        {n: round(float(w), 4) for n, w in zip(CLASS_NAMES, weights)},
    )

    return {
        "radius": radius,
        "scaler": scaler,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "train_loader": make_loader(x_train, y_train, shuffle=True),
        "val_loader": make_loader(x_val, y_val, shuffle=False),
        "test_loader": make_loader(x_test, y_test, shuffle=False),
        "weights": weights,
        "counts": {
            "train": kept_train,
            "validate": kept_val,
            "test": kept_test,
        },
        "raw_counts": {
            "train": raw_train,
            "validate": raw_val,
            "test": raw_test,
        },
    }
