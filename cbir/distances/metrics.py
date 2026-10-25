import numpy as np
from typing import Callable, List

def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance."""
    return float(np.sum(np.abs(a - b)))

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""
    return float(np.sqrt(np.sum((a - b) ** 2)))

def minkowski_distance(a: np.ndarray, b: np.ndarray, p: int = 3) -> float:
    """Minkowski distance."""
    return float(np.sum(np.abs(a - b) ** p) ** (1/p))

def chi_squared_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Chi-squared distance for histograms."""
    return float(0.5 * np.sum(((a - b) ** 2) / (a + b + 1e-10)))

def histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection (returns SIMILARITY)."""
    return float(np.sum(np.minimum(a, b)))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot_product / (norm_a * norm_b + 1e-10))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine similarity)."""
    return 1.0 - cosine_similarity(a, b)

def bhattacharyya_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance for normalized histograms."""
    bc = np.sum(np.sqrt(a * b))
    return float(-np.log(bc + 1e-10))

_AVAILABLE_METRICS = {
    'l1': l1_distance,
    'l2': l2_distance,
    'minkowski': minkowski_distance,
    'chi_squared': chi_squared_distance,
    'intersection': histogram_intersection,
    'cosine_sim': cosine_similarity,
    'cosine_dist': cosine_distance,
    'bhattacharyya': bhattacharyya_distance
}

def get_distance_function(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Factory that returns the function by name."""
    func = _AVAILABLE_METRICS.get(name.lower())
    if func is None:
        raise ValueError(f"Unknown metric: {name}")
    return func

def get_available_metrics() -> List[str]:
    """Returns list of available metric names."""
    return list(_AVAILABLE_METRICS.keys())
