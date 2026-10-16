import json
d=json.load(open("lit/scan_s2.json",encoding="utf-8"))
key=["multi-fidelity natural convection surrogate",
 "surface radiation natural convection square cavity benchmark",
 "low fidelity correction high fidelity thermal radiation",
 "multi-fidelity radiation heat transfer",
 "radiation natural convection reduced-order model"]
for q in key:
    r=d[q].get("data") or []
    print(f"\n=== {q}  total={d[q].get('total')} ===")
    for p in r[:18]:
        ex=p.get("externalIds") or {}
        print(f"  {p.get('year')} | {str(p.get('venue'))[:36]:36s} | {str(p.get('title'))[:86]}")
        if ex.get("DOI"): print(f"        DOI:{ex['DOI']}")
