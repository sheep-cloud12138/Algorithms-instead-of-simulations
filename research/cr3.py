import urllib.request, urllib.parse, json, re, time, os
OUT = "research/lit"; os.makedirs(OUT, exist_ok=True)
def cr(q, rows=10):
    p = {"query.bibliographic": q, "rows": rows,
         "select": "DOI,title,container-title,issued,abstract,type"}
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers={"User-Agent":"lit-review/1.0 (mailto:a@b.c)"})
    return json.load(urllib.request.urlopen(req, timeout=50))["message"]["items"]

QUERIES = [
 ("J-rad","deep learning surrogate model surface radiation coupled natural convection enclosure"),
 ("J-rad","machine learning surrogate radiative natural convection heat transfer"),
 ("J-rad","neural network prediction heat transfer surface radiation buoyancy cavity"),
 ("J-nb","non-Boussinesq natural convection deep learning correction"),
 ("J-nb","compressible effects natural convection surrogate model neural network"),
 ("J-nb","variable properties natural convection machine learning surrogate"),
 ("J-mf","multi-fidelity surrogate model heat transfer CFD correction"),
 ("J-mf","physics-based correction machine learning turbulence closure review"),
 ("J-mf","low fidelity high fidelity correction surrogate convection"),
 ("J-scale","scale extrapolation surrogate heat transfer different geometric scale"),
 ("J-scale","geometry size generalization neural network heat transfer"),
 ("J-scale","dimensional analysis machine learning heat transfer surrogate"),
 ("J-unit","units equivariance machine learning dimensional consistency"),
 ("J-unit","dimensionless group neural network operator learning PDE"),
 ("J-brk","similarity breaking correction convection surrogate"),
]
lines = []
for tag, q in QUERIES:
    lines.append("="*110); lines.append(f"[{tag}] CROSSREF: {q}")
    for attempt in range(3):
        try:
            items = cr(q); break
        except Exception as e:
            items = [{"title":[f"FAIL {e}"],"DOI":"","container-title":[""],"issued":{}}]
            time.sleep(5)
    for it in items:
        t = (it.get("title") or ["?"])[0]
        j = (it.get("container-title") or ["?"])[0] if it.get("container-title") else "?"
        y = it.get("issued",{}).get("date-parts",[[None]])[0][0]
        lines.append(f"  - {it.get('DOI')} | {y} | {j} | {it.get('type')}")
        lines.append(f"      {t}")
        a = it.get("abstract")
        if a:
            a = " ".join(re.sub(r"<[^>]+>"," ",a).split())
            lines.append(f"      ABS: {a[:420]}")
    open(os.path.join(OUT,"crossref.txt"),"w",encoding="utf-8").write("\n".join(lines))
    print(f"[{tag}] {q} -> {len(items)}", flush=True)
    time.sleep(1.0)
print("DONE")
