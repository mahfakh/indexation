"""Quick end-to-end API test."""
import requests
import json

BASE = "http://localhost:8000"

# 1. Datasets
r = requests.get(f"{BASE}/api/datasets")
datasets = r.json()
print("=== DATASETS ===")
for ds in datasets:
    name = ds["name"]
    total = ds["total_images"]
    cats = ds["categories_count"]
    cov = ds["descriptor_coverage"]
    print(f"  {name}: {total} images, {cats} categories")
    print(f"  Descriptors: {cov}")

# 2. Text search
r = requests.post(f"{BASE}/api/search/text", json={"query": "butterfly", "max_results": 5})
res = r.json()
print(f"\n=== TEXT SEARCH: 'butterfly' ===")
print(f"  {res['total']} results in {res['time_ms']:.1f}ms")
for hit in res["results"][:3]:
    print(f"  #{hit['rank']} {hit['category']} ({hit['image_id'][:12]}...)")

# 3. Visual search — VP-Tree
if res["results"]:
    query_id = res["results"][0]["image_id"]
    r2 = requests.post(f"{BASE}/api/search/image", json={
        "query_image_id": query_id,
        "descriptor_type": "color_rgb",
        "distance_metric": "l2",
        "max_results": 10,
        "use_index": True,
    })
    vis = r2.json()
    print(f"\n=== VISUAL SEARCH (color_rgb, VP-Tree) ===")
    print(f"  Query: {query_id[:12]}... (butterfly)")
    print(f"  {vis['total']} results in {vis['time_ms']:.1f}ms")
    print(f"  Search type: {vis['search_type']}, comparisons: {vis['comparisons']}")
    for hit in vis["results"][:5]:
        print(f"  #{hit['rank']} {hit['category']} dist={hit['distance']:.4f}")

    # 4. Same but exhaustive
    r3 = requests.post(f"{BASE}/api/search/image", json={
        "query_image_id": query_id,
        "descriptor_type": "color_rgb",
        "distance_metric": "l2",
        "max_results": 10,
        "use_index": False,
    })
    exh = r3.json()
    print(f"\n=== VISUAL SEARCH (color_rgb, EXHAUSTIVE) ===")
    print(f"  {exh['total']} results in {exh['time_ms']:.1f}ms")
    print(f"  Search type: {exh['search_type']}, comparisons: {exh['comparisons']}")
    for hit in exh["results"][:5]:
        print(f"  #{hit['rank']} {hit['category']} dist={hit['distance']:.4f}")

# 5. Available descriptors & metrics
r4 = requests.get(f"{BASE}/api/descriptors/types")
r5 = requests.get(f"{BASE}/api/metrics")
print(f"\n=== AVAILABLE ===")
print(f"  Descriptors: {r4.json()}")
print(f"  Metrics: {r5.json()}")

print("\n=== ALL E2E TESTS PASSED ===")
