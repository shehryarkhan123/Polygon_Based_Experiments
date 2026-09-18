"""Add GNDVI, EVI, and SAVI next to B4 on every train/validate/test polygon CSV.

Band map for the 4-band ENVI stack:
  B1 = Blue, B2 = Green, B3 = Red, B4 = NIR

DN values in this dataset are ~170-450, so they are converted to 0-1
reflectance with /1000 before EVI and SAVI (those formulas use +1 and L=0.5).
GNDVI is a ratio, so the scale cancels; it is still computed on reflectance.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "validate", "test")
CLASSES = ("cpf_poly", "sl_poly", "water_poly", "urban_poly", "road_poly")
BANDS = ("B1", "B2", "B3", "B4")
NEW_COLS = ("GNDVI", "EVI", "SAVI")
REFLECTANCE_SCALE = 1000.0
EPS = 1e-8


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    missing = [b for b in BANDS if b not in df.columns]
    if missing:
        raise KeyError(f"missing {missing}")

    blue = df["B1"].to_numpy(dtype=np.float64) / REFLECTANCE_SCALE
    green = df["B2"].to_numpy(dtype=np.float64) / REFLECTANCE_SCALE
    red = df["B3"].to_numpy(dtype=np.float64) / REFLECTANCE_SCALE
    nir = df["B4"].to_numpy(dtype=np.float64) / REFLECTANCE_SCALE

    gndvi = (nir - green) / (nir + green + EPS)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + EPS)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + EPS)

    out = df.drop(columns=[c for c in NEW_COLS if c in df.columns], errors="ignore")
    b4_at = out.columns.get_loc("B4") + 1
    out.insert(b4_at, "GNDVI", gndvi)
    out.insert(b4_at + 1, "EVI", evi)
    out.insert(b4_at + 2, "SAVI", savi)
    return out


def process_file(path: Path) -> int:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    n = len(df)
    add_indices(df).to_csv(path, index=False)
    return n


def main() -> None:
    total_files = 0
    total_rows = 0
    for class_dir in CLASSES:
        for split in SPLITS:
            folder = ROOT / class_dir / split
            files = sorted(folder.glob("Polygon_*.csv"))
            if not files:
                print(f"skip (no CSVs): {class_dir}/{split}")
                continue
            for path in files:
                n = process_file(path)
                total_files += 1
                total_rows += n
                print(f"updated: {class_dir}/{split}/{path.name} ({n} pixels)")
    print(
        f"\nDone. Added GNDVI, EVI, SAVI to {total_files} files ({total_rows:,} pixels). "
        "Features are now B1, B2, B3, B4, GNDVI, EVI, SAVI."
    )


if __name__ == "__main__":
    main()
