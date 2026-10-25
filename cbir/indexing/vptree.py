import numpy as np
import random
import pickle
import time
import logging
import heapq
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class _Node:
    point: np.ndarray
    point_id: str
    mu: float
    left: Optional['_Node'] = None
    right: Optional['_Node'] = None

class VPTree:
    def __init__(self, distance_fn: Callable):
        self.distance_fn = distance_fn
        self.root = None
        self._size = 0

    @property
    def size(self):
        return self._size

    @property
    def is_built(self):
        return self.root is not None

    def build(self, points: List[Tuple[str, np.ndarray]]) -> float:
        start_time = time.time()
        self._size = len(points)
        items = list(points)
        self.root = self._build_recursive(items)
        build_time = time.time() - start_time
        logger.info(f"VP-Tree built with {self._size} items in {build_time:.4f} seconds")
        return build_time

    def _build_recursive(self, items: List[Tuple[str, np.ndarray]]) -> Optional[_Node]:
        if not items:
            return None
            
        vp_idx = random.randrange(len(items))
        vp_id, vp_point = items.pop(vp_idx)
        
        if not items:
            return _Node(point=vp_point, point_id=vp_id, mu=0.0)
            
        distances = [(self.distance_fn(vp_point, p[1]), p) for p in items]
        distances.sort(key=lambda x: x[0])
        
        median_idx = len(distances) // 2
        mu = distances[median_idx][0]
        
        left_items = [p for d, p in distances[:median_idx]]
        right_items = [p for d, p in distances[median_idx:]]
        
        node = _Node(point=vp_point, point_id=vp_id, mu=mu)
        node.left = self._build_recursive(left_items)
        node.right = self._build_recursive(right_items)
        
        return node

    def query(self, target: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        if not self.root:
            return []
            
        heap = [] # max-heap to store (negative_distance, index, point_id, distance)
        # We need index to resolve tie-breakers in heap operations when distances are equal
        comparisons_counter = [0]
        self._query_counter = 0 # tie-breaker
        
        self._query_recursive(self.root, target, k, heap, comparisons_counter)
        
        results = [(item[2], item[3]) for item in heap]
        results.sort(key=lambda x: x[1])
        return results

    def _query_recursive(self, node: _Node, target: np.ndarray, k: int, heap: list, comparisons_counter: list):
        if not node:
            return
            
        dist = self.distance_fn(target, node.point)
        comparisons_counter[0] += 1
        
        tau = -heap[0][0] if len(heap) == k else float('inf')
        
        if dist < tau:
            self._query_counter += 1
            if len(heap) == k:
                heapq.heappushpop(heap, (-dist, self._query_counter, node.point_id, dist))
            else:
                heapq.heappush(heap, (-dist, self._query_counter, node.point_id, dist))
            tau = -heap[0][0] if len(heap) == k else float('inf')
            
        if dist < node.mu:
            self._query_recursive(node.left, target, k, heap, comparisons_counter)
            if dist + tau >= node.mu:
                self._query_recursive(node.right, target, k, heap, comparisons_counter)
        else:
            self._query_recursive(node.right, target, k, heap, comparisons_counter)
            if dist - tau <= node.mu:
                self._query_recursive(node.left, target, k, heap, comparisons_counter)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'VPTree':
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_stats(self) -> dict:
        depths = []
        def _get_depths(node, current_depth):
            if not node:
                return
            if not node.left and not node.right:
                depths.append(current_depth)
            _get_depths(node.left, current_depth + 1)
            _get_depths(node.right, current_depth + 1)
            
        _get_depths(self.root, 0)
        
        return {
            'size': self._size,
            'max_depth': max(depths) if depths else 0,
            'min_depth': min(depths) if depths else 0,
            'avg_depth': sum(depths)/len(depths) if depths else 0
        }
