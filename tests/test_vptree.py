"""
Unit tests for the VP-Tree index.

Run with:  pytest tests/test_vptree.py -v
"""

import os
import tempfile
import numpy as np
import pytest
from cbir.indexing.vptree import VPTree
from cbir.distances.metrics import l2_distance


def _make_points(n: int = 50, dim: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype(np.float32)
    return [(f"img_{i:04d}", vecs[i]) for i in range(n)]


class TestVPTreeBuild:
    def test_builds_without_error(self):
        tree = VPTree(l2_distance)
        points = _make_points(20)
        build_time = tree.build(points)
        assert tree.is_built
        assert tree.size == 20
        assert build_time >= 0.0

    def test_single_point(self):
        tree = VPTree(l2_distance)
        vec = np.array([0.5, 0.5], dtype=np.float32)
        tree.build([("only", vec)])
        assert tree.is_built
        results = tree.query(vec, k=1)
        assert len(results) == 1
        assert results[0][0] == "only"

    def test_empty_build(self):
        tree = VPTree(l2_distance)
        tree.build([])
        assert not tree.is_built
        results = tree.query(np.zeros(5), k=5)
        assert results == []


class TestVPTreeQuery:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.points = _make_points(100, dim=8)
        self.tree = VPTree(l2_distance)
        self.tree.build(self.points)
        self.all_vecs = {pid: vec for pid, vec in self.points}

    def _brute_force(self, query, k):
        dists = [(pid, l2_distance(query, vec)) for pid, vec in self.points]
        dists.sort(key=lambda x: x[1])
        return dists[:k]

    def test_nearest_neighbour_correctness(self):
        """VP-Tree top-1 must match brute-force top-1."""
        query = np.random.rand(8).astype(np.float32)
        tree_res = self.tree.query(query, k=1)
        bf_res = self._brute_force(query, k=1)
        assert tree_res[0][0] == bf_res[0][0], (
            f"VP-Tree found {tree_res[0][0]} but brute-force found {bf_res[0][0]}"
        )

    def test_top_k_ids_match_brute_force(self):
        """Top-10 results must be identical to brute-force."""
        query = np.random.rand(8).astype(np.float32)
        k = 10
        tree_ids = {r[0] for r in self.tree.query(query, k=k)}
        bf_ids = {r[0] for r in self._brute_force(query, k=k)}
        assert tree_ids == bf_ids

    def test_results_sorted_ascending(self):
        query = np.random.rand(8).astype(np.float32)
        results = self.tree.query(query, k=10)
        dists = [r[1] for r in results]
        assert dists == sorted(dists)

    def test_k_larger_than_dataset(self):
        query = np.random.rand(8).astype(np.float32)
        results = self.tree.query(query, k=200)
        assert len(results) == 100  # only 100 points in tree


class TestVPTreePersistence:
    def test_save_and_load(self):
        points = _make_points(30, dim=6)
        tree = VPTree(l2_distance)
        tree.build(points)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            path = tmp.name

        try:
            tree.save(path)
            loaded = VPTree.load(path)
            query = np.random.rand(6).astype(np.float32)
            orig_res = tree.query(query, k=5)
            load_res = loaded.query(query, k=5)
            assert [r[0] for r in orig_res] == [r[0] for r in load_res]
        finally:
            os.unlink(path)


class TestVPTreeStats:
    def test_get_stats_returns_dict(self):
        tree = VPTree(l2_distance)
        tree.build(_make_points(20))
        stats = tree.get_stats()
        assert "size" in stats
        assert "max_depth" in stats
        assert stats["size"] == 20
