import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
def inv2abs(inv):
    if not inv: return None
    pos={}
    for w,ps in inv.items():
        for p in ps: pos[p]=w
    return " ".join(pos[k] for k in sorted(pos))
def oa(q,filt=None,n=8,sel="id,doi,title,publication_year,primary_location,abstract_inverted_index,cited_by_count,authorships"):
    f=f"title_and_abstract.search:{q}"
    if filt: f+=","+filt
    r=requests.get("https://api.openalex.org/works",headers=H,timeout=30,params={"filter":f,"per-page":n,"select":sel})
    return r.json()
for q in ["Benchmark solution of 3D natural convection flows with surface radiation",
          "dimensionless solution radiation turbulent natural convection square rectangular cavities",
          "Scale analysis combined radiation convection"]:
    d=oa(q)
    print(f"\n##### {q}  count={d['meta']['count']}")
    for w in d["results"]:
        src=(w.get("primary_location") or {}).get("source") or {}
        print(f"  {w['publication_year']} | {str(src.get('display_name'))[:40]:40s} | {str(w['title'])[:88]}")
        print(f"        DOI:{w.get('doi')} cites={w.get('cited_by_count')}")
        a=inv2abs(w.get("abstract_inverted_index"))
        if a: print(f"        A: {a[:520]}")
    time.sleep(0.5)
