import json
d=json.load(open("lit/scan_s2b.json",encoding="utf-8"))
for q,v in d.items():
    if v.get("total",0)==0: continue
    print(f"\n=== {q}  total={v.get('total')} ===")
    for p in (v.get("data") or [])[:14]:
        ex=p.get("externalIds") or {}
        print(f"  {p.get('year')} | {str(p.get('venue'))[:34]:34s} | {str(p.get('title'))[:86]}")
        if ex.get("DOI"): print(f"        DOI:{ex['DOI']}")
