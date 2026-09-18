from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPATIAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SPATIAL_DIR / "results"
ABLATION_DIR = SPATIAL_DIR / "ablation"
ABLATION_RESULTS_DIR = ABLATION_DIR / "results"

BANDS = ["B1", "B2", "B3", "B4"]
CLASS_NAMES = ["CPF 253", "SL 284", "water", "urban", "road"]
N_CLASSES = len(CLASS_NAMES)
N_FEATURES = len(BANDS)
XY_COLS = ["File X", "File Y"]


def shifts_for_radius(radius: int) -> tuple[tuple[int, int], ...]:
    """Row-wise (dx, dy) offsets. Centre is (0, 0)."""
    return tuple(
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    )


def n_neighbors_for_radius(radius: int) -> int:
    return (2 * radius + 1) ** 2


def center_index_for_radius(radius: int) -> int:
    side = 2 * radius + 1
    return radius * side + radius


# Default paper method: 3x3. Centre is token index 4.
WINDOW_RADIUS = 1
SHIFTS = shifts_for_radius(WINDOW_RADIUS)
N_NEIGHBORS = n_neighbors_for_radius(WINDOW_RADIUS)
CENTER_INDEX = center_index_for_radius(WINDOW_RADIUS)

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

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 1
FF_DIM = 64

PERM_REPEATS = 10
PERM_SAMPLE_SIZE = 20000

RF_N_ESTIMATORS_GRID = (100, 200)
RF_MAX_DEPTH_GRID = (20, None)
