"""
CLI script pour évaluer les performances du moteur CBIR.

Usage :
    python scripts/run_evaluation.py --dataset objects --descriptor color_rgb --metric l2 --k 20
    python scripts/run_evaluation.py --dataset objects --compare-descriptors --k 10
"""

import argparse
import logging
import sys
import os
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbir.config import settings
from cbir.database import DatabaseManager
from cbir.indexing.indexer import ImageIndexer
from cbir.search import SearchEngine


def precision_at_k(retrieved_ids: list, query_category: str,
                   all_images_by_id: dict, k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant = sum(
        1 for img_id in top_k
        if all_images_by_id.get(img_id, {}).get("category") == query_category
    )
    return relevant / len(top_k)


def recall_at_k(retrieved_ids: list, query_category: str,
                all_images_by_id: dict, total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = sum(
        1 for img_id in top_k
        if all_images_by_id.get(img_id, {}).get("category") == query_category
    )
    return relevant / total_relevant


def average_precision(retrieved_ids: list, query_category: str,
                      all_images_by_id: dict) -> float:
    hits = 0
    sum_precisions = 0.0
    for i, img_id in enumerate(retrieved_ids):
        if all_images_by_id.get(img_id, {}).get("category") == query_category:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / hits if hits > 0 else 0.0


def evaluate(
    db: DatabaseManager,
    indexer: ImageIndexer,
    engine: SearchEngine,
    dataset: str,
    descriptor_type: str,
    distance_metric: str,
    k: int,
    num_queries: int,
    use_index: bool,
    logger,
) -> dict:
    # Collect all indexed images for this dataset
    rows = db.get_images_with_descriptor(descriptor_type, dataset)
    if not rows:
        logger.error(f"No images with descriptor '{descriptor_type}' for dataset '{dataset}'.")
        return {}

    all_images_by_id = {r["image_id"]: r for r in rows}

    # Pick random query images
    query_pool = random.sample(list(all_images_by_id.values()),
                                min(num_queries, len(all_images_by_id)))

    precisions, recalls, avg_precs, latencies, comparisons_list = [], [], [], [], []
    per_category: dict = {}

    for i, query_img in enumerate(query_pool):
        q_id = query_img["image_id"]
        q_cat = query_img.get("category", "")

        # Total relevant images for this category (excluding query itself)
        total_relevant = sum(
            1 for r in all_images_by_id.values()
            if r.get("category") == q_cat and r["image_id"] != q_id
        )

        try:
            res = engine.visual_search(
                query_image_id=q_id,
                descriptor_type=descriptor_type,
                distance_metric=distance_metric,
                max_results=k + 1,
                dataset=dataset,
                use_index=use_index,
            )
        except Exception as exc:
            logger.warning(f"Search failed for {q_id}: {exc}")
            continue

        retrieved_ids = [r["image_id"] for r in res["results"] if r["image_id"] != q_id][:k]

        p = precision_at_k(retrieved_ids, q_cat, all_images_by_id, k)
        r = recall_at_k(retrieved_ids, q_cat, all_images_by_id, total_relevant, k)
        ap = average_precision(retrieved_ids, q_cat, all_images_by_id)

        precisions.append(p)
        recalls.append(r)
        avg_precs.append(ap)
        latencies.append(res["time_ms"])
        comparisons_list.append(res["comparisons"])

        if q_cat not in per_category:
            per_category[q_cat] = {"precisions": [], "avg_precs": []}
        per_category[q_cat]["precisions"].append(p)
        per_category[q_cat]["avg_precs"].append(ap)

        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1}/{len(query_pool)} queries …")

    n = len(precisions)
    if n == 0:
        return {}

    cat_summary = {
        cat: {
            "precision": sum(v["precisions"]) / len(v["precisions"]),
            "map": sum(v["avg_precs"]) / len(v["avg_precs"]),
            "queries": len(v["precisions"]),
        }
        for cat, v in per_category.items()
    }

    return {
        "descriptor_type": descriptor_type,
        "distance_metric": distance_metric,
        "k": k,
        "num_queries": n,
        "use_index": use_index,
        "precision_at_k": round(sum(precisions) / n, 4),
        "recall_at_k": round(sum(recalls) / n, 4),
        "map": round(sum(avg_precs) / n, 4),
        "avg_latency_ms": round(sum(latencies) / n, 2),
        "avg_comparisons": round(sum(comparisons_list) / n, 1),
        "per_category": cat_summary,
    }


def print_report(results: list, compare_mode: bool):
    print("\n" + "=" * 70)
    print("  CBIR EVALUATION REPORT")
    print("=" * 70)

    if compare_mode:
        # Table de comparaison
        print(f"\n{'Descripteur':<15} {'Métrique':<12} {'P@K':>7} {'R@K':>7} {'mAP':>7} {'Lat(ms)':>9} {'Compar.':>9}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['descriptor_type']:<15} {r['distance_metric']:<12} "
                f"{r['precision_at_k']:>7.4f} {r['recall_at_k']:>7.4f} "
                f"{r['map']:>7.4f} {r['avg_latency_ms']:>9.1f} "
                f"{r['avg_comparisons']:>9.0f}"
            )
    else:
        r = results[0]
        print(f"\nDescripteur     : {r['descriptor_type']}")
        print(f"Métrique        : {r['distance_metric']}")
        print(f"k               : {r['k']}")
        print(f"Requêtes        : {r['num_queries']}")
        print(f"Mode index      : {'VP-Tree' if r['use_index'] else 'Exhaustif'}")
        print(f"\nPrécision@{r['k']:>2}   : {r['precision_at_k']:.4f}")
        print(f"Rappel@{r['k']:>2}      : {r['recall_at_k']:.4f}")
        print(f"mAP             : {r['map']:.4f}")
        print(f"Latence moy.    : {r['avg_latency_ms']:.1f} ms")
        print(f"Comparaisons    : {r['avg_comparisons']:.0f}")

        # Top 5 catégories par mAP
        cats = sorted(r["per_category"].items(),
                      key=lambda x: x[1]["map"], reverse=True)[:5]
        if cats:
            print(f"\nTop 5 catégories (mAP):")
            for cat, v in cats:
                print(f"  {cat:<25} mAP={v['map']:.4f}  P@k={v['precision']:.4f}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CBIR Evaluation Tool")
    parser.add_argument("--dataset", default="objects",
                        choices=["objects", "textures"])
    parser.add_argument("--descriptor", default="color_rgb",
                        choices=["color_rgb", "color_hsv", "lbp", "hog"])
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--queries", type=int, default=50,
                        help="Nombre de requêtes à évaluer.")
    parser.add_argument("--compare-descriptors", action="store_true",
                        help="Compare tous les descripteurs disponibles.")
    parser.add_argument("--compare-index", action="store_true",
                        help="Compare VP-Tree vs recherche exhaustive.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)

    db = DatabaseManager(settings.DB_PATH)
    indexer = ImageIndexer(db, cache_dir=settings.FEATURES_CACHE_DIR)
    engine = SearchEngine(db, indexer)

    all_results = []

    if args.compare_descriptors:
        descriptors = ["color_rgb", "color_hsv", "lbp", "hog"]
        for desc in descriptors:
            logger.info(f"Evaluating {desc} …")
            res = evaluate(db, indexer, engine, args.dataset, desc,
                           args.metric, args.k, args.queries, True, logger)
            if res:
                all_results.append(res)
        print_report(all_results, compare_mode=True)

    elif args.compare_index:
        for use_idx in [True, False]:
            label = "VP-Tree" if use_idx else "Exhaustif"
            logger.info(f"Evaluating with {label} …")
            res = evaluate(db, indexer, engine, args.dataset, args.descriptor,
                           args.metric, args.k, args.queries, use_idx, logger)
            if res:
                all_results.append(res)
        print_report(all_results, compare_mode=True)

    else:
        res = evaluate(db, indexer, engine, args.dataset, args.descriptor,
                       args.metric, args.k, args.queries, True, logger)
        if res:
            all_results.append(res)
            print_report(all_results, compare_mode=False)

    db.close()


if __name__ == "__main__":
    main()
