import requests, json, time
H={"User-Agent":"lit-scan/1.0"}
def s2(q,extra=""):
    for a in range(5):
        try:
            r=requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",headers=H,timeout=45,
              params={"query":q,"fields":"title,year,venue,externalIds","year":extra} if extra else
                     {"query":q,"fields":"title,year,venue,externalIds"})
            if r.status_code==200: return r.json()
            if r.status_code==429: time.sleep(6+a*6); continue
            return {"err":r.status_code}
        except Exception: time.sleep(4)
    return {"err":"fail"}
QS=[("artificial neural network radiation parameter natural convection enclosure",None),
    ("ANN surrogate radiation parameter buoyancy driven cavity",None),
    ("machine learning prediction natural convection thermal radiation nanofluid",None),
    ("artificial neural network heat transfer thermal radiation natural convection cavity",None),
    ("artificial neural network radiation parameter natural convection","2023-2026"),
    ("surrogate model natural convection radiation enclosure","2023-2026")]
out={}
for q,yr in QS:
    d=s2(q,yr)
    print(f"\n### total={d.get('total')} err={d.get('err')} | {q} [{yr or 'all'}]")
    for p in (d.get("data") or [])[:10]:
        ex=p.get("externalIds") or {}
        print(f"  {p.get('year')} | {str(p.get('venue'))[:34]:34s} | {str(p.get('title'))[:84]}")
        if ex.get("DOI"): print(f"        DOI:{ex['DOI']}")
    out[f"{q}|{yr}"]=d; time.sleep(3.5)
json.dump(out,open("lit/scan_s2d.json","w",encoding="utf-8"),ensure_ascii=False)
print("\nsaved")
