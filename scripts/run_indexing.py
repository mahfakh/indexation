"""
CLI script to index a dataset and optionally build a VP-Tree.

Usage examples:
    # Index the Objects dataset with colour + LBP descriptors
    python scripts/run_indexing.py --dataset objects --descriptors color_rgb lbp hog

    # Force re-index and build VP-Tree afterward
    python scripts/run_indexing.py --dataset objects --descriptors color_rgb --force --build-index

    # Index Textures dataset
    python scripts/run_indexing.py --dataset textures --descriptors color_rgb lbp hog
"""

import argparse
import logging
import sys
import os
import time

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbir.config import settings
from cbir.database import DatabaseManager
from cbir.descriptors import DESCRIPTOR_TYPES
from cbir.indexing.indexer import ImageIndexer


def main():
    parser = argparse.ArgumentParser(description="CBIR Dataset Indexer")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["objects", "textures"],
        help="Dataset to index.",
    )
    parser.add_argument(
        "--descriptors",
        nargs="+",
        default=["color_rgb"],
        choices=DESCRIPTOR_TYPES,
        help=f"Descriptor types to extract. Available: {DESCRIPTOR_TYPES}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if descriptors already cached.",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build VP-Tree index after extraction.",
    )
    parser.add_argument(
        "--metric",
        default="l2",
        help="Distance metric for VP-Tree (default: l2).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Resolve dataset path
    path_map = {
        "objects": settings.DATASET_B_OBJECTS_PATH,
        "textures": settings.DATASET_B_TEXTURES_PATH,
    }
    image_dir = path_map[args.dataset]

    if not os.path.isdir(image_dir):
        logger.error(
            f"Dataset directory not found: {image_dir}\n"
            "Please check the DATASET_B_OBJECTS_PATH / DATASET_B_TEXTURES_PATH "
            "settings in your .env file."
        )
        sys.exit(1)

    logger.info(f"Dataset      : {args.dataset}")
    logger.info(f"Image dir    : {image_dir}")
    logger.info(f"Descriptors  : {args.descriptors}")
    logger.info(f"Force reindex: {args.force}")

    # Initialise services
    os.makedirs(settings.FEATURES_CACHE_DIR, exist_ok=True)
    os.makedirs(settings.INDEX_DIR, exist_ok=True)
    db = DatabaseManager(settings.DB_PATH)
    indexer = ImageIndexer(db, cache_dir=settings.FEATURES_CACHE_DIR)

    # Run indexing
    t0 = time.time()
    stats = indexer.index_dataset(
        dataset_name=args.dataset,
        image_dir=image_dir,
        descriptor_types=args.descriptors,
        force_reindex=args.force,
    )
    elapsed = time.time() - t0

    logger.info("=" * 50)
    logger.info("Indexing complete")
    logger.info(f"  Total images  : {stats['total']}")
    logger.info(f"  Processed     : {stats['processed']}")
    logger.info(f"  Errors        : {stats['errors']}")
    logger.info(f"  Time          : {elapsed:.1f}s")
    logger.info("=" * 50)

    if args.build_index:
        for desc_type in args.descriptors:
            logger.info(f"Building VP-Tree for {desc_type} / {args.metric} …")
            try:
                tree = indexer.build_vptree_index(
                    descriptor_type=desc_type,
                    distance_metric=args.metric,
                    dataset=args.dataset,
                )
                tree_stats = tree.get_stats()
                logger.info(
                    f"  VP-Tree built: {tree_stats['size']} items, "
                    f"depth {tree_stats['min_depth']}–{tree_stats['max_depth']}"
                )
            except Exception as exc:
                logger.error(f"  Failed to build VP-Tree for {desc_type}: {exc}")

    db.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
