"""Test fuzzy search."""
import requests

BASE = "http://localhost:8000"

# Exact search
r = requests.post(f"{BASE}/api/search/text",
                  json={"query": "butterfly", "max_results": 3, "fuzzy_threshold": 0.0})
res = r.json()
total = res["total"]
fuzzy = res["fuzzy"]
print(f"EXACT  'butterfly' -> {total} results, fuzzy={fuzzy}")

# Fuzzy with typo: buterfly -> butterfly
r2 = requests.post(f"{BASE}/api/search/text",
                   json={"query": "buterfly", "max_results": 3, "fuzzy_threshold": 0.7})
res2 = r2.json()
print(f"FUZZY  'buterfly'  -> {res2['total']} results, fuzzy={res2['fuzzy']}")
for hit in res2["results"][:3]:
    print(f"  #{hit['rank']} {hit['category']}  score={hit['score']}  ({hit['match_type']})")

# Fuzzy with truncation: airplan -> airplanes
r3 = requests.post(f"{BASE}/api/search/text",
                   json={"query": "airplan", "max_results": 3, "fuzzy_threshold": 0.7})
res3 = r3.json()
print(f"FUZZY  'airplan'   -> {res3['total']} results")
for hit in res3["results"][:3]:
    print(f"  #{hit['rank']} {hit['category']}  score={hit['score']}")

# Fuzzy with accent: motorbike -> Motorbikes
r4 = requests.post(f"{BASE}/api/search/text",
                   json={"query": "motorbik", "max_results": 3, "fuzzy_threshold": 0.7})
res4 = r4.json()
print(f"FUZZY  'motorbik'  -> {res4['total']} results")
