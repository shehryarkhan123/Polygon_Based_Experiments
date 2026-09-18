import shutil
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
INIT_DIR = ROOT / "init_dataset"

# Whole-polygon split: ~70% train / 15% validate / 15% test by polygon count.
CLASSES = {
    "water": {
        "csv": INIT_DIR / "water.csv",
        "n_clusters": 33,
        "n_train": 23,
        "n_validate": 5,
        "n_test": 5,
        "out_dir": ROOT / "water_poly",
        "readme": ROOT / "water_readme.md",
        "lat_expected": (31.35059145, 31.42369429),
        "lon_expected": (73.42515779, 73.50928382),
    },
    "urban": {
        "csv": INIT_DIR / "urban.csv",
        "n_clusters": 45,
        "n_train": 31,
        "n_validate": 7,
        "n_test": 7,
        "out_dir": ROOT / "urban_poly",
        "readme": ROOT / "urban_readme.md",
        "lat_expected": (31.38062559, 31.42014691),
        "lon_expected": (73.44539945, 73.49051373),
    },
    "road": {
        "csv": INIT_DIR / "road.csv",
        "n_clusters": 12,
        "n_train": 8,
        "n_validate": 2,
        "n_test": 2,
        "out_dir": ROOT / "road_poly",
        "readme": ROOT / "road_readme.md",
        "lat_expected": (31.34255964, 31.40953181),
        "lon_expected": (73.41053294, 73.50314023),
    },
}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment=";", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    missing = [c for c in ("Lat", "Lon") if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing {missing}; got {list(df.columns)}")
    return df


def assign_splits(sizes: dict[int, int], n_train: int, n_val: int, n_test: int) -> dict[str, list[int]]:
    """Greedy fill so pixel shares stay near 70/15/15. Whole polygons only."""
    targets = {"train": 0.70, "validate": 0.15, "test": 0.15}
    slots = {"train": n_train, "validate": n_val, "test": n_test}
    assigned = {"train": [], "validate": [], "test": []}
    pixels = {"train": 0, "validate": 0, "test": 0}
    total = sum(sizes.values()) or 1

    for pid, size in sorted(sizes.items(), key=lambda item: -item[1]):
        candidates = [k for k, cap in slots.items() if len(assigned[k]) < cap]
        if not candidates:
            raise RuntimeError("split slots exhausted")
        best = min(candidates, key=lambda k: (pixels[k] / total) / targets[k])
        assigned[best].append(pid)
        pixels[best] += size

    for key in assigned:
        assigned[key] = sorted(assigned[key])
    return assigned


def copy_split(out_dir: Path, prefix: str, split: dict[str, list[int]]) -> None:
    for subset, ids in split.items():
        dest = out_dir / subset
        dest.mkdir(parents=True, exist_ok=True)
        for i in ids:
            name = f"Polygon_{prefix}_{i}.csv"
            src = out_dir / name
            shutil.copy2(src, dest / name)
            print(f"copied: {out_dir.name}/{subset}/{name}")


def write_readme(
    name: str,
    cfg: dict,
    n_pixels: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    sizes: dict[int, int],
    split: dict[str, list[int]],
) -> None:
    rows = "\n".join(
        f"| `Polygon_{name}_{i}.csv` | {sizes[i]:,} |"
        for i in range(1, cfg["n_clusters"] + 1)
    )
    split_rows = []
    for subset in ("train", "validate", "test"):
        ids = split[subset]
        pix = sum(sizes[i] for i in ids)
        share = 100.0 * pix / n_pixels
        id_txt = ", ".join(str(i) for i in ids)
        split_rows.append(
            f"| `{name}_poly/{subset}/` | {id_txt} | {pix:,} | {share:.1f}% |"
        )
    expected_lat = cfg["lat_expected"]
    expected_lon = cfg["lon_expected"]
    text = f"""# {name.upper()} polygon split

Source file: `init_dataset/{name}.csv` ({n_pixels:,} ENVI pixel rows).
Script: `create_polygone_water_urban_road.py`.

Lat range: {lat_min:.8f} to {lat_max:.8f} (expected {expected_lat[0]} to {expected_lat[1]})
Lon range: {lon_min:.8f} to {lon_max:.8f} (expected {expected_lon[0]} to {expected_lon[1]})

## How the {cfg['n_clusters']} polygons were split

All {cfg['n_clusters']} {name} polygons were exported together in one CSV. Coordinates sit in a tight box, so sorting by `Lat`/`Lon` mixes pixels from different shapes.

Split method:

1. Read the combined CSV and strip header spaces so columns resolve as `Lat` and `Lon`.
2. Standardize `Lat` and `Lon` so neither axis dominates distance.
3. Run **KMeans** (`n_clusters={cfg['n_clusters']}`, `random_state=42`, `n_init=10`) on those coordinates.
4. Write each cluster to its own CSV. The cluster ID is not kept in the output.
5. Copy whole polygons into train / validate / test (~70 / 15 / 15 by polygon count, balanced by pixels).

Each output file has the original columns: `File X`, `File Y`, `Map X`, `Map Y`, `Lat`, `Lon`, `B1`, `B2`, `B3`, `B4`.

## Pixel count per polygon

| File | Pixels |
|---|---|
{rows}
| **Total** | **{n_pixels:,}** |

## Train / validate / test

Whole polygons only (no random pixel split).

| Set | Polygons | Pixels | Share |
|---|---|---|---|
{chr(10).join(split_rows)}
| **Total** | **{cfg['n_clusters']}** | **{n_pixels:,}** | **100%** |
"""
    cfg["readme"].write_text(text, encoding="utf-8")
    print(f"wrote {cfg['readme'].name}")


def process_class(name: str, cfg: dict) -> None:
    print(f"\n=== {name.upper()} ===")
    df = load_csv(cfg["csv"])
    print(f"Loaded {len(df):,} pixels")
    print(f"Columns: {list(df.columns)}")
    lat_min, lat_max = float(df["Lat"].min()), float(df["Lat"].max())
    lon_min, lon_max = float(df["Lon"].min()), float(df["Lon"].max())
    print(f"Lat range: {lat_min:.8f} to {lat_max:.8f}")
    print(f"Lon range: {lon_min:.8f} to {lon_max:.8f}")

    n = cfg["n_clusters"]
    print(f"Grouping pixels into {n} independent polygons...")
    coords = StandardScaler().fit_transform(df[["Lat", "Lon"]])
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    df["Polygon_ID"] = kmeans.fit_predict(coords)

    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for i in range(n):
        polygon_df = df[df["Polygon_ID"] == i].drop(columns=["Polygon_ID"])
        sizes[i + 1] = len(polygon_df)
        path = out_dir / f"Polygon_{name}_{i + 1}.csv"
        polygon_df.to_csv(path, index=False)
        print(f"saved: {out_dir.name}/{path.name} ({len(polygon_df)} pixels)")

    split = assign_splits(sizes, cfg["n_train"], cfg["n_validate"], cfg["n_test"])
    copy_split(out_dir, name, split)
    write_readme(name, cfg, len(df), lat_min, lat_max, lon_min, lon_max, sizes, split)


def main() -> None:
    for name, cfg in CLASSES.items():
        process_class(name, cfg)
    print("\nSuccess! Water, urban, and road polygons are clustered and split.")


if __name__ == "__main__":
    main()
