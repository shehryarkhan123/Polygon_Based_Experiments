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
)


def _read_split_csvs(class_dir: Path, split: str, label: int) -> pd.DataFrame:
    folder = class_dir / split
    files = sorted(folder.glob("Polygon_*.csv"))
    if not files:
        raise FileNotFoundError(f"No polygon CSVs in {folder}")
    frames = []
    for path in files:
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        missing = [b for b in BANDS if b not in df.columns]
        if missing:
            raise KeyError(f"{path.name} missing columns {missing}")
        part = df[list(BANDS)].copy()
        part["label"] = label
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    frames = [
        _read_split_csvs(CLASS_DIRS[name], split, i)
        for i, name in enumerate(CLASS_NAMES)
    ]
    data = pd.concat(frames, ignore_index=True)
    x = data[list(BANDS)].to_numpy(dtype=np.float32)
    y = data["label"].to_numpy(dtype=np.int64)
    return x, y


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


def prepare_data() -> dict:
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("validate")
    x_test, y_test = load_split("test")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    weights = class_weights(y_train)
    print("class weights:", {n: round(float(w), 4) for n, w in zip(CLASS_NAMES, weights)})

    return {
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
            "train": class_counts(y_train),
            "validate": class_counts(y_val),
            "test": class_counts(y_test),
        },
    }
