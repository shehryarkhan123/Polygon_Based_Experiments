import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Polygon-level split (~70% / 15% / 15% by pixels). Whole polygons stay in one set.
CPF_SPLIT = {
    "train": [1, 2, 3, 6, 8, 9, 11, 12, 13, 14, 15],
    "validate": [4, 5],
    "test": [7, 10],
}
SL_SPLIT = {
    "train": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17],
    "validate": [1, 5, 16],
    "test": [4, 14],
}


def copy_split(src_dir: Path, prefix: str, split: dict) -> None:
    for subset, ids in split.items():
        dest = src_dir / subset
        dest.mkdir(parents=True, exist_ok=True)
        for i in ids:
            name = f"Polygon_{prefix}_{i}.csv"
            src = src_dir / name
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dest / name)
            print(f"copied: {src_dir.name}/{subset}/{name}")


def main() -> None:
    copy_split(ROOT / "cpf_poly", "cpf", CPF_SPLIT)
    copy_split(ROOT / "sl_poly", "sl", SL_SPLIT)
    print("\nDone. Train / validate / test folders are ready.")


if __name__ == "__main__":
    main()
