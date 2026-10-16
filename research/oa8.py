import urllib.request, urllib.parse, json, os, time, re
OUT="research/lit"; os.makedirs(OUT, exist_ok=True)
def oa(q, n=10):
    p = {"search": q, "per-page": n, "mailto":"lit@example.com",
         "select":"id,doi,title,publication_year,primary_location,abstract_inverted_index,type"}
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers={"User-Agent":"lit-review/1.0 (mailto:lit@example.com)"})
    return json.load(urllib.request.urlopen(req, timeout=50))["results"]
def absn(w):
    ii = w.get("abstract_inverted_index")
    if not ii: return ""
    pos = {}
    for word, idxs in ii.items():
        for i in idxs: pos[i] = word
    return " ".join(pos[k] for k in sorted(pos))
qs = [
 ("Q3","deep learning surrogate model surface radiation coupled natural convection enclosure"),
 ("Q3","machine learning surrogate model thermal radiation natural convection heat transfer"),
 ("Q2","non-Boussinesq correction machine learning natural convection surrogate"),
 ("Q2","Boussinesq approximation breakdown correction neural network convection"),
 ("Q1","dimensional analysis neural network heat transfer dimensionless groups"),
 ("Q1","units equivariance machine learning dimensional consistency physics"),
 ("Q4","scale extrapolation geometric size generalization surrogate heat transfer"),
 ("Q4","neural network extrapolation different geometry size convection"),
 ("Q5","multi-fidelity correction machine learning heat transfer convection"),
 ("Q5","low fidelity solver plus learned correction CFD surrogate"),
]
lines=[]
for tag,q in qs:
    lines.append("="*110); lines.append(f"[{tag}] OPENALEX: {q}")
    try: res = oa(q)
    except Exception as e:
        lines.append(f"  FAIL {e}"); res=[]
    for w in res:
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        lines.append(f"  - {w.get('doi')} | {w.get('publication_year')} | {src} | {w.get('type')} | cites={w.get('cited_by_count')}")
        lines.append(f"      {w.get('title')}")
        a = absn(w)
        if a: lines.append(f"      ABS: {a[:400]}")
    open(os.path.join(OUT,"openalex.txt"),"w",encoding="utf-8").write("\n".join(lines))
    print(f"[{tag}] {q} -> {len(res)}", flush=True)
    time.sleep(1)
print("DONE")
