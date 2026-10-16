import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
DOIS=["10.1017/jfm.2020.395","10.1016/J.ENCONMAN.2013.01.024",
 "10.1016/J.IJHEATMASSTRANSFER.2016.11.061","10.1016/J.IJHEATMASSTRANSFER.2006.07.022",
 "10.1016/j.ijthermalsci.2020.106310","10.2172/1566085","10.1016/j.fusengdes.2026.115789",
 "10.1080/10407782.2023.2272792","10.3390/e23070901","10.1108/hff-01-2026-0110",
 "10.1016/j.energy.2020.118543","10.1016/J.PNUCENE.2019.01.017","10.1016/J.NUCENGDES.2019.03.028"]
def oa_doi(doi):
    r=requests.get(f"https://api.openalex.org/works/https://doi.org/{doi}",headers=H,timeout=30)
    return r.json() if r.status_code==200 else {"err":r.status_code}
def inv2abs(inv):
    if not inv: return None
    pos={}
    for w,ps in inv.items():
        for p in ps: pos[p]=w
    return " ".join(pos[k] for k in sorted(pos))
out={}
for d in DOIS:
    w=oa_doi(d)
    if "err" in w: print(f"\n### {d}\n  ERR {w['err']}"); out[d]=w; time.sleep(0.5); continue
    src=(w.get("primary_location") or {}).get("source") or {}
    a=inv2abs(w.get("abstract_inverted_index"))
    print(f"\n### {d}\n  {w.get('publication_year')} | {src.get('display_name')} | cites={w.get('cited_by_count')}")
    print(f"  T: {w.get('title')}")
    print(f"  A: {(a[:700] if a else '(OpenAlex 无摘要)')}")
    out[d]=w; time.sleep(0.6)
json.dump(out,open("lit/abstracts_key.json","w",encoding="utf-8"),ensure_ascii=False)
print("\nsaved lit/abstracts_key.json")
