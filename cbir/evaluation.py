import time
from typing import List, Dict
from cbir.search import SearchEngine
from cbir.database import DatabaseManager

class Evaluator:
    def __init__(self, search_engine: SearchEngine, db: DatabaseManager):
        self.search_engine = search_engine
        self.db = db
        
    def _is_relevant(self, query_id: str, doc_id: str) -> bool:
        q_img = self.db.get_image(query_id)
        d_img = self.db.get_image(doc_id)
        if not q_img or not d_img:
            return False
        return q_img.get("category") == d_img.get("category")

    def precision_at_k(self, query_id: str, retrieved_ids: List[str], k: int) -> float:
        if not retrieved_ids or k <= 0:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_count = sum(1 for doc_id in top_k if self._is_relevant(query_id, doc_id))
        return relevant_count / k
        
    def recall_at_k(self, query_id: str, retrieved_ids: List[str], k: int) -> float:
        q_img = self.db.get_image(query_id)
        if not q_img:
            return 0.0
            
        category = q_img.get("category")
        all_images = self.db.get_all_images()
        total_relevant = sum(1 for img in all_images if img.get("category") == category)
        
        if total_relevant == 0:
            return 0.0
            
        top_k = retrieved_ids[:k]
        relevant_count = sum(1 for doc_id in top_k if self._is_relevant(query_id, doc_id))
        return relevant_count / total_relevant

    def average_precision(self, query_id: str, retrieved_ids: List[str]) -> float:
        if not retrieved_ids:
            return 0.0
            
        relevant_count = 0
        sum_precisions = 0.0
        
        for i, doc_id in enumerate(retrieved_ids):
            if self._is_relevant(query_id, doc_id):
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                sum_precisions += precision_at_i
                
        if relevant_count == 0:
            return 0.0
            
        return sum_precisions / relevant_count

    def mean_average_precision(self, query_ids: List[str], descriptor_type: str, distance_metric: str, k: int = 20) -> Dict:
        ap_sum = 0.0
        prec_sum = 0.0
        rec_sum = 0.0
        time_sum = 0.0
        per_category = {}
        
        for qid in query_ids:
            res = self.search_engine.visual_search(
                query_image_id=qid,
                descriptor_type=descriptor_type,
                distance_metric=distance_metric,
                max_results=k,
                use_index=False
            )
            
            retrieved = [r["image_id"] for r in res["results"]]
            time_sum += res["time_ms"]
            
            ap = self.average_precision(qid, retrieved)
            prec = self.precision_at_k(qid, retrieved, k)
            rec = self.recall_at_k(qid, retrieved, k)
            
            ap_sum += ap
            prec_sum += prec
            rec_sum += rec
            
            q_img = self.db.get_image(qid)
            cat = q_img.get("category", "unknown") if q_img else "unknown"
            
            if cat not in per_category:
                per_category[cat] = {"ap_sum": 0.0, "count": 0}
            
            per_category[cat]["ap_sum"] += ap
            per_category[cat]["count"] += 1
            
        n = len(query_ids)
        if n == 0:
            return {}
            
        cat_stats = {cat: data["ap_sum"] / data["count"] for cat, data in per_category.items()}
            
        return {
            "map": ap_sum / n,
            "avg_precision_at_k": prec_sum / n,
            "avg_recall_at_k": rec_sum / n,
            "per_category": cat_stats,
            "num_queries": n,
            "avg_time_ms": time_sum / n
        }

    def compare_descriptors(self, query_ids: List[str], descriptor_types: List[str], distance_metric: str, k: int = 20) -> Dict:
        results = {}
        for desc in descriptor_types:
            stats = self.mean_average_precision(query_ids, desc, distance_metric, k)
            results[desc] = stats
        return results
        
    def compare_search_methods(self, query_ids: List[str], descriptor_type: str, distance_metric: str, k: int = 20) -> Dict:
        exhaustive_times = []
        indexed_times = []
        overlaps = []
        
        for qid in query_ids:
            res_ex = self.search_engine.visual_search(
                query_image_id=qid,
                descriptor_type=descriptor_type,
                distance_metric=distance_metric,
                max_results=k,
                use_index=False
            )
            
            res_ix = self.search_engine.visual_search(
                query_image_id=qid,
                descriptor_type=descriptor_type,
                distance_metric=distance_metric,
                max_results=k,
                use_index=True
            )
            
            exhaustive_times.append(res_ex["time_ms"])
            indexed_times.append(res_ix["time_ms"])
            
            ex_ids = set(r["image_id"] for r in res_ex["results"])
            ix_ids = set(r["image_id"] for r in res_ix["results"])
            
            if ex_ids:
                overlap = len(ex_ids.intersection(ix_ids)) / len(ex_ids)
                overlaps.append(overlap)
                
        return {
            "avg_exhaustive_time_ms": sum(exhaustive_times) / len(query_ids) if query_ids else 0,
            "avg_indexed_time_ms": sum(indexed_times) / len(query_ids) if query_ids else 0,
            "avg_overlap": sum(overlaps) / len(overlaps) if overlaps else 0
        }

    def generate_report(self, results: Dict) -> str:
        md = "## Descriptor Comparison Report\n\n"
        md += "| Descriptor | MAP | Precision@K | Recall@K | Avg Time (ms) |\n"
        md += "|---|---|---|---|---|\n"
        
        for desc, stats in results.items():
            if "map" in stats:
                md += f"| {desc} | {stats['map']:.4f} | {stats['avg_precision_at_k']:.4f} | {stats['avg_recall_at_k']:.4f} | {stats['avg_time_ms']:.2f} |\n"
                
        return md
