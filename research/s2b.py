import urllib.request, urllib.parse, json, time, os
OUT="research/lit"
def s2bulk(q, fields="title,year,venue,abstract,externalIds,citationCount", extra=None):
    p={"query":q,"fields":fields}
    if extra: p.update(extra)
    url="https://api.semanticscholar.org/graph/v1/paper/search/bulk?"+urllib.parse.urlencode(p)
    req=urllib.request.Request(url, headers={"User-Agent":"lit-review/1.0"})
    for _ in range(5):
        try: return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            print("  retry", e, flush=True); time.sleep(12)
    return {"data":[],"total":-1}
QS=[
 ("Q3-rad-conv-ML", '"surface radiation" + ("natural convection" | "buoyancy") + ("machine learning" | "neural network" | "surrogate" | "deep learning")'),
 ("Q3-surfrad-surrogate", '"surface radiation" + surrogate + ("convection" | "heat transfer")'),
 ("Q2-nonboussinesq", '"non-Boussinesq" + ("machine learning" | "neural" | "surrogate" | "data-driven")'),
 ("Q2-boussinesq-correction", 'Boussinesq + ("low-fidelity" | "reduced model") + correction + ("neural" | "machine learning")'),
 ("Q1-units-equiv", '"units equivariance" | "dimensional equivariance" | "unit equivariance"'),
 ("Q1-dim-analysis-ML", '"dimensional analysis" + ("neural operator" | "operator learning" | "deep learning") + ("PDE" | "field prediction")'),
 ("Q4-scale-extrap", '("scale extrapolation" | "size extrapolation" | "geometric scaling") + ("surrogate" | "neural network") + ("heat transfer" | "convection" | "fluid")'),
 ("Q5-analytic-base-corr", '("analytical model" | "physics-based baseline" | "low-fidelity model") + correction + ("neural network" | "machine learning" | "multi-fidelity") + ("heat transfer" | "convection")'),
 ("Q6-similarity-break", '("similarity breaking" | "breaks similarity" | "departure from similarity") + ("correction" | "surrogate" | "machine learning")'),
]
lines=[]
for tag,q in QS:
    lines.append("="*110); lines.append(f"[{tag}] S2BULK: {q}")
    d=s2bulk(q)
    lines.append(f"  total={d.get('total')}")
    for w in (d.get("data") or [])[:15]:
        ex=w.get("externalIds") or {}
        lines.append(f"  - {ex.get('DOI') or ex.get('ArXiv')} | {w.get('year')} | {w.get('venue')} | cites={w.get('citationCount')}")
        lines.append(f"      {w.get('title')}")
        a=w.get("abstract")
        if a: lines.append(f"      ABS: {' '.join(a.split())[:380]}")
    open(os.path.join(OUT,"s2.txt"),"w",encoding="utf-8").write("\n".join(lines))
    print(f"[{tag}] total={d.get('total')}", flush=True)
    time.sleep(4)
print("DONE")
