from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CNN_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CNN_DIR / "results"

BANDS = ["B1", "B2", "B3", "B4", "GNDVI", "EVI", "SAVI"]
CLASS_NAMES = ["CPF 253", "SL 284", "water", "urban", "road"]
N_CLASSES = len(CLASS_NAMES)
N_FEATURES = len(BANDS)

CLASS_DIRS = {
    "CPF 253": ROOT / "cpf_poly",
    "SL 284": ROOT / "sl_poly",
    "water": ROOT / "water_poly",
    "urban": ROOT / "urban_poly",
    "road": ROOT / "road_poly",
}

SEED = 42
BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
PATIENCE = 8
PERM_REPEATS = 10
PERM_SAMPLE_SIZE = 20000
