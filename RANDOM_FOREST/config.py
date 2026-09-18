from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RF_DIR = Path(__file__).resolve().parent
RESULTS_DIR = RF_DIR / "results"

BANDS = ["B1", "B2", "B3", "B4", "GNDVI", "EVI", "SAVI"]
CLASS_NAMES = ["CPF 253", "SL 284", "water", "urban", "road"]
N_CLASSES = len(CLASS_NAMES)

CLASS_DIRS = {
    "CPF 253": ROOT / "cpf_poly",
    "SL 284": ROOT / "sl_poly",
    "water": ROOT / "water_poly",
    "urban": ROOT / "urban_poly",
    "road": ROOT / "road_poly",
}

SEED = 42
N_ESTIMATORS_GRID = (100, 200)
MAX_DEPTH_GRID = (20, None)
PERM_REPEATS = 10
PERM_SAMPLE_SIZE = 20000
