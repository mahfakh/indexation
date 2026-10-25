"""
Unit tests for distance metrics.

Run with:  pytest tests/test_distances.py -v
"""

import numpy as np
import pytest
from cbir.distances.metrics import (
    l1_distance,
    l2_distance,
    minkowski_distance,
    chi_squared_distance,
    histogram_intersection,
    cosine_similarity,
    cosine_distance,
    bhattacharyya_distance,
    get_distance_function,
    get_available_metrics,
)


class TestL1Distance:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert l1_distance(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert l1_distance(a, b) == pytest.approx(7.0)

    def test_symmetry(self):
        a = np.random.rand(10)
        b = np.random.rand(10)
        assert l1_distance(a, b) == pytest.approx(l1_distance(b, a))


class TestL2Distance:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert l2_distance(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert l2_distance(a, b) == pytest.approx(5.0)

    def test_symmetry(self):
        a = np.random.rand(10)
        b = np.random.rand(10)
        assert l2_distance(a, b) == pytest.approx(l2_distance(b, a))


class TestMinkowski:
    def test_p1_equals_l1(self):
        a = np.random.rand(8)
        b = np.random.rand(8)
        assert minkowski_distance(a, b, p=1) == pytest.approx(l1_distance(a, b))

    def test_p2_equals_l2(self):
        a = np.random.rand(8)
        b = np.random.rand(8)
        assert minkowski_distance(a, b, p=2) == pytest.approx(l2_distance(a, b))


class TestChiSquared:
    def test_identical_histograms(self):
        h = np.array([0.25, 0.25, 0.25, 0.25])
        assert chi_squared_distance(h, h) == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        a = np.random.rand(16)
        b = np.random.rand(16)
        a /= a.sum()
        b /= b.sum()
        assert chi_squared_distance(a, b) >= 0.0


class TestHistogramIntersection:
    def test_identical_unit_histograms(self):
        h = np.array([0.25, 0.25, 0.25, 0.25])
        assert histogram_intersection(h, h) == pytest.approx(1.0)

    def test_orthogonal_histograms(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert histogram_intersection(a, b) == pytest.approx(0.0)

    def test_range_0_to_1_for_normalized(self):
        a = np.random.rand(8)
        b = np.random.rand(8)
        a /= a.sum()
        b /= b.sum()
        result = histogram_intersection(a, b)
        assert 0.0 <= result <= 1.0 + 1e-6


class TestCosine:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_distance_is_complement(self):
        a = np.random.rand(12)
        b = np.random.rand(12)
        assert cosine_distance(a, b) == pytest.approx(1.0 - cosine_similarity(a, b))


class TestBhattacharyya:
    def test_identical_histograms(self):
        h = np.array([0.25, 0.25, 0.25, 0.25])
        assert bhattacharyya_distance(h, h) == pytest.approx(0.0, abs=1e-4)

    def test_non_negative(self):
        a = np.random.rand(16)
        b = np.random.rand(16)
        a /= a.sum()
        b /= b.sum()
        assert bhattacharyya_distance(a, b) >= 0.0


class TestFactory:
    def test_get_known_metric(self):
        fn = get_distance_function("l2")
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert fn(a, b) == pytest.approx(5.0)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            get_distance_function("nonexistent")

    def test_available_metrics_is_list(self):
        metrics = get_available_metrics()
        assert isinstance(metrics, list)
        assert "l1" in metrics
        assert "l2" in metrics
