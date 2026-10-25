"""Quick URL verification."""
import requests

r = requests.post("http://localhost:8000/api/search/text",
                  json={"query": "butterfly", "max_results": 2})
results = r.json()["results"]
for hit in results:
    url = hit["url"]
    print(f"URL: {url}")
    r2 = requests.get(url)
    ct = r2.headers.get("content-type", "?")
    print(f"  -> status={r2.status_code}, type={ct}, size={len(r2.content)} bytes")
    print()
