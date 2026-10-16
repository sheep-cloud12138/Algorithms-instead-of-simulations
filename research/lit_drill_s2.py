import json
d=json.load(open("lit/scan_s2.json",encoding="utf-8"))
key=["surface radiation natural convection surrogate",
 "surface radiation natural convection machine learning",
 "radiative heat transfer natural convection neural network",
 "radiation coupled natural convection data-driven model",
 "multi-fidelity natural convection surrogate",
 "surface radiation natural convection square cavity benchmark",
 "low fidelity correction high fidelity thermal radiation",
 "multi-fidelity radiation heat transfer"]
for q in key:
    r=d[q].get("data") or []
    print(f"\n=== {q}  total={d[q].get('total')} ===")
    for p in r[:15]:
        ex=p.get("externalIds") or {}
        doi=ex.get("DOI","")
        print(f"  {p.get('year')} | {str(p.get('venue'))[:40]:40s} | {str(p.get('title'))[:88]}")
        if doi: print(f"        DOI:{doi}")
