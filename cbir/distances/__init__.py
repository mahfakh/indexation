"""
Distance metrics package — re-exports the main factory functions.
"""

from cbir.distances.metrics import get_distance_function as get_distance_fn
from cbir.distances.metrics import get_available_metrics

__all__ = ["get_distance_fn", "get_available_metrics"]
