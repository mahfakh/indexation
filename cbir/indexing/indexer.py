"""
Image indexing pipeline.

Responsibilities:
  - Scan dataset directories for images.
  - Register image metadata in SQLite via DatabaseManager.
  - Extract visual descriptors and cache them as .npy files.
  - Build and persist VP-Tree indexes for fast nearest-neighbour retrieval.
  - Support incremental re-indexing (skip already-processed images).
"""

import os
import time
import logging
import glob
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from cbir.database import DatabaseManager
from cbir.indexing.vptree import VPTree
from cbir.descriptors import get_extractor, DESCRIPTOR_TYPES
from cbir.distances import get_distance_fn

logger = logging.getLogger(__name__)

# Where VP-Tree pickle files are stored (relative to cache_dir)
_TREE_SUFFIX = ".vptree.pkl"


class ImageIndexer:
    """Orchestrates dataset scanning, feature extraction, and index construction."""

    def __init__(self, db: DatabaseManager, cache_dir: str):
        self.db = db
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_dataset(
        self,
        dataset_name: str,
        image_dir: str,
        descriptor_types: List[str],
        force_reindex: bool = False,
    ) -> Dict:
        """Scan *image_dir* recursively, register images and extract descriptors.

        Args:
            dataset_name: Logical name stored in the DB ('objects', 'textures').
            image_dir:    Root directory containing images.
            descriptor_types: List of descriptor names to extract per image.
            force_reindex: When True, re-extract even if already cached.

        Returns:
            Dict with keys 'total', 'processed', 'errors', 'time_seconds'.
        """
        self.db.update_indexing_status(
            dataset_name,
            status="running",
            started_at=datetime.now().isoformat(),
            processed_images=0,
            errors=0,
        )

        # Collect all image files
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
        image_files: List[str] = []
        for ext in extensions:
            image_files.extend(
                glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
            )
        image_files = sorted(set(image_files))

        total = len(image_files)
        self.db.update_indexing_status(dataset_name, total_images=total)
        logger.info(f"Found {total} images for dataset '{dataset_name}'")

        stats = {"total": total, "processed": 0, "errors": 0, "time_seconds": 0.0}
        start_time = time.time()

        for i, filepath in enumerate(image_files):
            try:
                image_id = self._get_image_id(filepath, dataset_name)
                # Category = parent directory name (works for Objects & Textures layouts)
                category = os.path.basename(os.path.dirname(filepath))
                filename = os.path.basename(filepath)
                file_size = os.path.getsize(filepath)

                if force_reindex or not self.db.image_exists(image_id):
                    self.db.add_image(
                        image_id, filepath, dataset_name,
                        category, filename, None, None, file_size,
                    )

                for desc_type in descriptor_types:
                    if not force_reindex and self.db.descriptor_exists(image_id, desc_type):
                        continue

                    feature = self._extract_descriptor(filepath, desc_type)
                    if feature is not None:
                        cache_path = os.path.join(
                            self.cache_dir, f"{image_id}_{desc_type}.npy"
                        )
                        np.save(cache_path, feature)
                        self.db.add_descriptor(
                            image_id, desc_type, cache_path, int(feature.shape[0])
                        )

                stats["processed"] += 1

            except Exception as exc:
                logger.error(f"Error processing {filepath}: {exc}")
                stats["errors"] += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{total} images")
                self.db.update_indexing_status(
                    dataset_name,
                    processed_images=stats["processed"],
                    errors=stats["errors"],
                )

        stats["time_seconds"] = time.time() - start_time
        self.db.update_indexing_status(
            dataset_name,
            status="completed",
            processed_images=stats["processed"],
            errors=stats["errors"],
            completed_at=datetime.now().isoformat(),
        )
        logger.info(
            f"Indexing done: {stats['processed']}/{total} images in "
            f"{stats['time_seconds']:.1f}s ({stats['errors']} errors)"
        )
        return stats

    def build_vptree_index(
        self,
        descriptor_type: str,
        distance_metric: str,
        dataset: Optional[str] = None,
    ) -> VPTree:
        """Load all cached features and build a VP-Tree index.

        Args:
            descriptor_type: E.g. 'color_rgb', 'lbp', 'hog'.
            distance_metric: E.g. 'l2', 'l1', 'chi_squared'.
            dataset: Optional dataset filter ('objects', 'textures').

        Returns:
            The built VPTree instance (also saved to disk).
        """
        ids, features = self.get_all_features(descriptor_type, dataset)
        if len(ids) == 0:
            raise ValueError(
                f"No features found for descriptor='{descriptor_type}' "
                f"dataset='{dataset}'. Run indexing first."
            )

        dist_fn = get_distance_fn(distance_metric)
        tree = VPTree(dist_fn)
        points = list(zip(ids, [features[i] for i in range(len(ids))]))
        build_time = tree.build(points)

        index_name = self._index_name(descriptor_type, distance_metric, dataset)
        tree_path = os.path.join(self.cache_dir, index_name + _TREE_SUFFIX)
        tree.save(tree_path)

        self.db.add_index_metadata(
            index_name, descriptor_type, distance_metric, len(ids), build_time
        )
        logger.info(
            f"VP-Tree '{index_name}' built ({len(ids)} items, {build_time:.2f}s)"
        )
        return tree

    def load_vptree_index(
        self,
        descriptor_type: str,
        distance_metric: str,
        dataset: Optional[str] = None,
    ) -> Optional[VPTree]:
        """Load a previously saved VP-Tree from disk.

        Returns:
            VPTree instance or None if not found / corrupted.
        """
        index_name = self._index_name(descriptor_type, distance_metric, dataset)
        tree_path = os.path.join(self.cache_dir, index_name + _TREE_SUFFIX)
        if os.path.exists(tree_path):
            try:
                return VPTree.load(tree_path)
            except Exception as exc:
                logger.error(f"Failed to load VP-Tree '{tree_path}': {exc}")
        return None

    def get_all_features(
        self, descriptor_type: str, dataset: Optional[str] = None
    ) -> Tuple[List[str], np.ndarray]:
        """Load all cached feature vectors for *descriptor_type*.

        Returns:
            Tuple (ids, feature_matrix) where feature_matrix has shape (N, D).
        """
        rows = self.db.get_images_with_descriptor(descriptor_type, dataset)
        ids: List[str] = []
        vectors: List[np.ndarray] = []

        for row in rows:
            cache_path = row.get("cache_path", "")
            if cache_path and os.path.exists(cache_path):
                try:
                    vec = np.load(cache_path)
                    ids.append(row["image_id"])
                    vectors.append(vec)
                except Exception as exc:
                    logger.warning(f"Could not load feature {cache_path}: {exc}")

        if not vectors:
            return [], np.empty((0,), dtype=np.float32)

        return ids, np.array(vectors, dtype=np.float32)

    def get_feature_for_image(
        self, image_id: str, descriptor_type: str
    ) -> Optional[np.ndarray]:
        """Load the cached feature vector for a single image.

        Returns:
            numpy array or None if not available.
        """
        cache_path = self.db.get_descriptor_path(image_id, descriptor_type)
        if cache_path and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception as exc:
                logger.warning(f"Could not load feature for {image_id}: {exc}")
        return None

    def extract_on_the_fly(
        self, image_path: str, descriptor_type: str
    ) -> Optional[np.ndarray]:
        """Extract a descriptor for an arbitrary (possibly external) image path."""
        return self._extract_descriptor(image_path, descriptor_type)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_descriptor(
        self, image_path: str, descriptor_type: str
    ) -> Optional[np.ndarray]:
        try:
            extractor = get_extractor(descriptor_type)
            return extractor(image_path)
        except Exception as exc:
            logger.error(
                f"Extraction failed [{descriptor_type}] {image_path}: {exc}"
            )
            return None

    @staticmethod
    def _get_image_id(file_path: str, dataset_name: str) -> str:
        """Generate a deterministic UUID from dataset name + absolute path."""
        unique_str = f"{dataset_name}:{os.path.abspath(file_path)}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_str))

    @staticmethod
    def _index_name(
        descriptor_type: str, distance_metric: str, dataset: Optional[str]
    ) -> str:
        name = f"vptree_{descriptor_type}_{distance_metric}"
        if dataset:
            name += f"_{dataset}"
        return name
