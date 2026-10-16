import urllib.request, urllib.parse, xml.etree.ElementTree as ET, json, sys, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
NS = {"a": "http://www.w3.org/2005/Atom"}

def arxiv(query, n=20, sort="relevance"):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": query, "start": 0, "max_results": n,
        "sortBy": sort, "sortOrder": "descending"})
    for _ in range(3):
        try:
            raw = urllib.request.urlopen(url, timeout=60).read(); break
        except Exception as e:
            print("retry", e, file=sys.stderr); time.sleep(3)
    else:
        return []
    root = ET.fromstring(raw)
    out = []
    for e in root.findall("a:entry", NS):
        out.append({
            "t": " ".join(e.find("a:title", NS).text.split()),
            "d": e.find("a:published", NS).text[:10],
            "id": e.find("a:id", NS).text.split("/abs/")[-1],
            "s": " ".join(e.find("a:summary", NS).text.split()),
        })
    return out

for q in json.loads(sys.argv[1]):
    print("=" * 95); print("Q:", q)
    for p in arxiv(q):
        print(f"[{p['d']}] {p['t']}\n     {p['id']}")
        print(f"     {p['s'][:260]}")
