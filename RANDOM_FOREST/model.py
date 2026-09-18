from sklearn.ensemble import RandomForestClassifier

from config import SEED


def make_rf(n_estimators: int, max_depth: int | None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
