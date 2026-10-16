import urllib.request, urllib.parse, json, os, time
OUT="research/lit"; os.makedirs(OUT, exist_ok=True)
def get(url, tries=6, wait=20):
    req = urllib.request.Request(url, headers={"User-Agent":"lit-review/1.0 (mailto:lit@example.com)"})
    for t in range(tries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            print("   retry", t, e, flush=True); time.sleep(wait)
    return None
def oa(q, n=8):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
        "filter": f"title_and_abstract.search:{q}",
        "per-page": n, "mailto": "lit@example.com",
        "sort": "relevance_score:desc"})
    return get(url)
def absn(w):
    ii = w.get("abstract_inverted_index")
    if not ii: return ""
    pos = {}
    for word, idxs in ii.items():
        for i in idxs: pos[i] = word
    return " ".join(pos[k] for k in sorted(pos))
qs = [
 ("Q3","surface radiation coupled natural convection machine learning surrogate"),
 ("Q3","thermal radiation natural convection heat transfer neural network prediction"),
 ("Q2","non-Boussinesq correction machine learning convection"),
 ("Q1","dimensional analysis neural network heat transfer dimensionless"),
 ("Q4","extrapolation surrogate model heat transfer geometric scale"),
 ("Q5","multi-fidelity correction machine learning heat transfer"),
]
lines=[]
for tag,q in qs:
    lines.append("="*110); lines.append(f"[{tag}] OPENALEX: {q}")
    d = oa(q)
    if d is None:
        lines.append("   (API FAILED)")
    else:
        lines.append(f"   count={d['meta']['count']}")
        for w in d["results"]:
            src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
            lines.append(f"  - {w.get('doi')} | {w.get('publication_year')} | {src} | cites={w.get('cited_by_count')}")
            lines.append(f"      {w.get('title')}")
            a = absn(w)
            if a: lines.append(f"      ABS: {a[:400]}")
    open(os.path.join(OUT,"openalex.txt"),"w",encoding="utf-8").write("\n".join(lines))
    print(f"[{tag}] {q} done", flush=True)
    time.sleep(2)
print("DONE")
