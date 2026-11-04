"""Pure set-to-set appearance and histogram comparisons."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalized_rows(values: ArrayLike) -> NDArray[np.float64]:
    """Normalize finite feature rows without dividing by zero."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite two-dimensional embedding matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def set_similarity(a: ArrayLike, b: ArrayLike, top_m: int) -> float:
    """Average top-m cross-crop cosine and normalized mean-vector cosine."""
    if top_m <= 0:
        raise ValueError("top_m must be positive")
    left, right = normalized_rows(a), normalized_rows(b)
    if left.shape[1] != right.shape[1]:
        raise ValueError("Embedding dimensions differ")
    if not len(left) or not len(right):
        return 0.0
    similarities = (left @ right.T).ravel()
    count = min(top_m, similarities.size)
    top = float(np.partition(similarities, -count)[-count:].mean())
    mean_a, mean_b = left.mean(axis=0), right.mean(axis=0)
    denominator = np.linalg.norm(mean_a) * np.linalg.norm(mean_b)
    mean_score = float(mean_a @ mean_b / denominator) if denominator else 0.0
    return float(np.clip((top + mean_score) / 2, 0, 1))


def color_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Return one minus Bhattacharyya distance for nonnegative histograms."""
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Histogram shapes differ")
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or (left < 0).any()
        or (right < 0).any()
    ):
        raise ValueError("Histogram values must be finite and nonnegative")
    if left.sum() == 0 or right.sum() == 0:
        return 0.0
    coefficient = np.sqrt(left / left.sum() * right / right.sum()).sum()
    return float(1 - np.sqrt(np.clip(1 - coefficient, 0, 1)))
