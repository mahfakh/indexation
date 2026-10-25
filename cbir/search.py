"""
Search engine — text search, visual (CBIR) search, and combined search.

All methods return plain dicts that are JSON-serializable and match the
FastAPI Pydantic response models defined in cbir/api.py.
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

from cbir.database import DatabaseManager
from cbir.distances.metrics import get_distance_function, get_available_metrics
from cbir.indexing.vptree import VPTree
from cbir.indexing.indexer import ImageIndexer

logger = logging.getLogger(__name__)


class SearchEngine:
    """High-level search interface wrapping text and visual retrieval."""

    def __init__(
        self,
        db: DatabaseManager,
        indexer: ImageIndexer,
        static_base_url: str = "/static",
    ):
        self.db = db
        self.indexer = indexer
        self.static_base_url = static_base_url
        # In-memory VP-Tree cache: key = (descriptor_type, distance_metric)
        self._tree_cache: Dict[Tuple[str, str], VPTree] = {}

    # ------------------------------------------------------------------
    # Text search
    # ------------------------------------------------------------------

    def text_search(
        self, query: str, dataset: Optional[str] = None,
        max_results: int = 50, fuzzy_threshold: float = 0.0,
    ) -> Dict:
        """Keyword-based search over category names, filenames, and image IDs.

        Multiple keywords are split by whitespace and AND-combined: all keywords
        must appear somewhere in the image's text fields.

        Args:
            query: Space-separated keywords.
            dataset: Optional dataset filter.
            max_results: Maximum number of results.
            fuzzy_threshold: 0.0 = exact substring match (default).
                             0.1–1.0 = fuzzy matching with SequenceMatcher.
                             Recommended: 0.6–0.8 for typo tolerance.

        Returns:
            Dict with keys: query, results, total, time_ms.
        """
        from difflib import SequenceMatcher

        start = time.time()
        keywords = [k.lower().strip() for k in query.split() if k.strip()]

        if not keywords:
            return {"query": query, "results": [], "total": 0,
                    "time_ms": 0.0}

        use_fuzzy = fuzzy_threshold > 0.0

        if use_fuzzy:
            # Fuzzy mode: scan all categories, then fetch matching images
            all_categories = self.db.get_all_categories(dataset)
            matching_cats = set()

            for cat in all_categories:
                cat_lower = cat.lower()
                for kw in keywords:
                    # Check substring first (fast path)
                    if kw in cat_lower:
                        matching_cats.add(cat)
                        break
                    # Fuzzy ratio on the whole category name
                    ratio = SequenceMatcher(None, kw, cat_lower).ratio()
                    if ratio >= fuzzy_threshold:
                        matching_cats.add(cat)
                        break
                    # Also check each word in the category name
                    for word in cat_lower.replace("_", " ").replace("-", " ").split():
                        ratio = SequenceMatcher(None, kw, word).ratio()
                        if ratio >= fuzzy_threshold:
                            matching_cats.add(cat)
                            break

            # Fetch all images from matching categories
            results = []
            for cat in matching_cats:
                imgs = self.db.get_images_by_category(cat)
                if dataset:
                    imgs = [i for i in imgs if i.get("dataset") == dataset]

                # Compute best fuzzy score for this category
                best_ratio = max(
                    max(
                        SequenceMatcher(None, kw, cat.lower()).ratio()
                        for kw in keywords
                    ),
                    1.0 if any(kw in cat.lower() for kw in keywords) else 0.0,
                )

                for img in imgs:
                    results.append({
                        "image_id": img["image_id"],
                        "file_path": img.get("file_path", ""),
                        "url": self._path_to_url(img.get("file_path", "")),
                        "category": img.get("category", ""),
                        "score": round(float(best_ratio), 4),
                        "match_type": "fuzzy" if best_ratio < 1.0 else "exact",
                        "distance": None,
                        "rank": 0,
                    })

        else:
            # Exact mode: use DB LIKE search as fast first pass
            rows = self.db.search_images_by_text(keywords[0])
            if dataset:
                rows = [r for r in rows if r.get("dataset") == dataset]

            results = []
            for img in rows:
                fields = " ".join([
                    (img.get("category") or "").lower(),
                    (img.get("filename") or "").lower(),
                    (img.get("image_id") or "").lower(),
                ])
                if all(kw in fields for kw in keywords):
                    score = sum(fields.count(kw) for kw in keywords) / len(keywords)
                    results.append({
                        "image_id": img["image_id"],
                        "file_path": img.get("file_path", ""),
                        "url": self._path_to_url(img.get("file_path", "")),
                        "category": img.get("category", ""),
                        "score": round(float(score), 4),
                        "match_type": "exact",
                        "distance": None,
                        "rank": 0,
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:max_results]
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "time_ms": round((time.time() - start) * 1000, 2),
            "fuzzy": use_fuzzy,
        }

    # ------------------------------------------------------------------
    # Visual (CBIR) search
    # ------------------------------------------------------------------

    def visual_search(
        self,
        query_image_id: Optional[str] = None,
        query_image_path: Optional[str] = None,
        query_features: Optional[np.ndarray] = None,
        descriptor_type: str = "color_rgb",
        distance_metric: str = "l2",
        max_results: int = 20,
        dataset: Optional[str] = None,
        use_index: bool = True,
    ) -> Dict:
        """Find the most visually similar images.

        Feature priority: query_features > query_image_id (load from cache)
                          > query_image_path (extract on the fly).

        Returns:
            Dict with keys: query_image_id, descriptor_type, distance_metric,
                            results, total, time_ms, search_type, comparisons.
        """
        start = time.time()

        # 1. Obtain query features
        if query_features is None:
            if query_image_id:
                query_features = self.indexer.get_feature_for_image(
                    query_image_id, descriptor_type
                )
            elif query_image_path:
                query_features = self.indexer.extract_on_the_fly(
                    query_image_path, descriptor_type
                )

        if query_features is None:
            raise ValueError(
                "Could not obtain query features. "
                "Provide query_image_id (indexed) or query_image_path."
            )

        # 2. Run search
        search_type = "exhaustive"
        comparisons = 0
        raw_results: List[Tuple[str, float]] = []

        if use_index:
            tree = self._get_or_load_tree(descriptor_type, distance_metric, dataset)
            if tree is not None and tree.is_built:
                try:
                    raw_results = tree.query(query_features, k=max_results)
                    comparisons = getattr(tree, "_last_comparisons", len(raw_results))
                    search_type = "indexed"
                except Exception as exc:
                    logger.warning(f"VP-Tree query failed ({exc}), falling back to exhaustive search.")

        if search_type == "exhaustive":
            raw_results, comparisons = self._exhaustive_search(
                query_features, descriptor_type, distance_metric, dataset, max_results
            )

        # 3. Format results (skip the query image itself if present)
        formatted: List[Dict] = []
        for rank, (img_id, distance) in enumerate(raw_results):
            if img_id == query_image_id:
                continue
            img = self.db.get_image(img_id)
            if img:
                formatted.append({
                    "image_id": img_id,
                    "file_path": img.get("file_path", ""),
                    "url": self._path_to_url(img.get("file_path", "")),
                    "category": img.get("category", ""),
                    "distance": round(float(distance), 6),
                    "score": None,
                    "rank": len(formatted) + 1,
                    "match_type": None,
                })
            if len(formatted) >= max_results:
                break

        return {
            "query_image_id": query_image_id,
            "descriptor_type": descriptor_type,
            "distance_metric": distance_metric,
            "results": formatted,
            "total": len(formatted),
            "time_ms": round((time.time() - start) * 1000, 2),
            "search_type": search_type,
            "comparisons": comparisons,
        }

    # ------------------------------------------------------------------
    # Combined search (text filter + visual ranking)
    # ------------------------------------------------------------------

    def combined_search(
        self,
        text_query: str,
        query_image_id: str,
        descriptor_type: str,
        distance_metric: str,
        text_weight: float = 0.5,
        max_results: int = 20,
    ) -> Dict:
        """Text-filtered visual search.

        1. Filter images by text query (keeps up to 1000).
        2. Rank them by visual distance to the query image.

        Returns:
            Same structure as visual_search.
        """
        start = time.time()

        text_res = self.text_search(text_query, max_results=1000)
        filtered_ids = [r["image_id"] for r in text_res["results"]]

        if not filtered_ids:
            return {
                "query_image_id": query_image_id,
                "text_query": text_query,
                "descriptor_type": descriptor_type,
                "distance_metric": distance_metric,
                "results": [],
                "total": 0,
                "time_ms": round((time.time() - start) * 1000, 2),
                "search_type": "combined_empty",
                "comparisons": 0,
            }

        query_features = self.indexer.get_feature_for_image(
            query_image_id, descriptor_type
        )
        if query_features is None:
            raise ValueError(f"No cached features for image '{query_image_id}'")

        dist_fn = get_distance_function(distance_metric)
        ranked: List[Tuple[str, float]] = []

        for img_id in filtered_ids:
            if img_id == query_image_id:
                continue
            feats = self.indexer.get_feature_for_image(img_id, descriptor_type)
            if feats is not None:
                ranked.append((img_id, float(dist_fn(query_features, feats))))

        ranked.sort(key=lambda x: x[1])
        ranked = ranked[:max_results]

        formatted: List[Dict] = []
        for rank, (img_id, distance) in enumerate(ranked):
            img = self.db.get_image(img_id)
            if img:
                formatted.append({
                    "image_id": img_id,
                    "file_path": img.get("file_path", ""),
                    "url": self._path_to_url(img.get("file_path", "")),
                    "category": img.get("category", ""),
                    "distance": round(distance, 6),
                    "score": None,
                    "rank": rank + 1,
                    "match_type": "combined",
                })

        return {
            "query_image_id": query_image_id,
            "text_query": text_query,
            "descriptor_type": descriptor_type,
            "distance_metric": distance_metric,
            "results": formatted,
            "total": len(formatted),
            "time_ms": round((time.time() - start) * 1000, 2),
            "search_type": "combined_filtered",
            "comparisons": len(ranked),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _exhaustive_search(
        self,
        query_features: np.ndarray,
        descriptor_type: str,
        distance_metric: str,
        dataset: Optional[str],
        max_results: int,
    ) -> Tuple[List[Tuple[str, float]], int]:
        """Brute-force linear scan over all cached features."""
        dist_fn = get_distance_function(distance_metric)
        ids, feature_matrix = self.indexer.get_all_features(descriptor_type, dataset)

        if len(ids) == 0:
            return [], 0

        distances = [
            (ids[i], float(dist_fn(query_features, feature_matrix[i])))
            for i in range(len(ids))
        ]
        distances.sort(key=lambda x: x[1])
        return distances[:max_results], len(ids)

    def _get_or_load_tree(
        self, descriptor_type: str, distance_metric: str, dataset: Optional[str]
    ) -> Optional[VPTree]:
        key = (descriptor_type, distance_metric, dataset or "")
        if key not in self._tree_cache:
            tree = self.indexer.load_vptree_index(
                descriptor_type, distance_metric, dataset
            )
            # If dataset=None, try known dataset-specific trees as fallback
            if tree is None and dataset is None:
                for ds in ("objects", "textures"):
                    tree = self.indexer.load_vptree_index(
                        descriptor_type, distance_metric, ds
                    )
                    if tree is not None:
                        break
            self._tree_cache[key] = tree  # type: ignore
        return self._tree_cache.get(key)

    def _path_to_url(self, file_path: str) -> str:
        """Convert an absolute filesystem path to a /static/... URL.

        The FastAPI static mount exposes ``settings.STATIC_DIR`` at ``/static``.
        We compute the path relative to STATIC_DIR so the URL resolves correctly.
        """
        if not file_path:
            return ""
        import os
        from cbir.config import settings

        # Normalise both paths for reliable comparison
        abs_path = os.path.normpath(os.path.abspath(file_path))
        static_root = os.path.normpath(os.path.abspath(settings.STATIC_DIR))

        try:
            rel = os.path.relpath(abs_path, static_root)
        except ValueError:
            # Different drives on Windows — fallback to basename
            rel = os.path.basename(file_path)

        # Use forward slashes for URL
        rel = rel.replace("\\", "/")
        return f"{self.static_base_url}/{rel}"
