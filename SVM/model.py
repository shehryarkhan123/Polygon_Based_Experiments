from sklearn.svm import LinearSVC

from config import MAX_ITER, SEED


def make_svm(C: float) -> LinearSVC:
    return LinearSVC(
        C=C,
        class_weight="balanced",
        dual="auto",
        max_iter=MAX_ITER,
        random_state=SEED,
    )
