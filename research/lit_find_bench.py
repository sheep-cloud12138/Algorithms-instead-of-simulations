import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
def s2q(q,n=8):
    for a in range(5):
        try:
            r=requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",headers=H,timeout=45,
              params={"query":q,"fields":"title,year,venue,externalIds,abstract"})
            if r.status_code==200: return r.json()
            if r.status_code==429: time.sleep(6+a*6); continue
            return {"err":r.status_code}
        except Exception: time.sleep(4)
    return {"err":"fail"}
QS=["Benchmark solution 3D natural convection flows surface radiation air-filled cavity",
 "overview improvement natural convection heat transfer surface thermal radiation",
 "Scale Analysis of Combined Radiation and Convection",
 "dimensionless solution radiation turbulent natural convection square rectangular cavities",
 "similarity law radiation natural convection enclosure scale"]
for q in QS:
    d=s2q(q)
    print(f"\n### {q}  total={d.get('total')} err={d.get('err')}")
    for p in (d.get("data") or [])[:6]:
        ex=p.get("externalIds") or {}
        print(f"  {p.get('year')} | {str(p.get('venue'))[:36]:36s} | {str(p.get('title'))[:90]}")
        if ex.get("DOI"): print(f"        DOI:{ex['DOI']}")
    time.sleep(3.5)
