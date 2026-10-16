import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
def inv2abs(inv):
    if not inv: return None
    pos={}
    for w,ps in inv.items():
        for p in ps: pos[p]=w
    return " ".join(pos[k] for k in sorted(pos))
for d in ["10.1016/j.ijheatmasstransfer.2026.129230","10.1016/j.icheatmasstransfer.2025.109805",
          "10.1016/j.icheatmasstransfer.2026.111511","10.1016/j.ijheatmasstransfer.2014.02.009",
          "10.1016/j.csite.2022.101902","10.1115/1.1677409","10.1115/imece2003-41874",
          "10.1016/j.rineng.2024.102514"]:
    r=requests.get(f"https://api.openalex.org/works/https://doi.org/{d}",headers=H,timeout=30)
    if r.status_code!=200: print(f"\n### {d} ERR {r.status_code}"); continue
    w=r.json(); src=(w.get("primary_location") or {}).get("source") or {}
    a=inv2abs(w.get("abstract_inverted_index"))
    print(f"\n### {d}\n  {w.get('publication_year')} | {src.get('display_name')} | cites={w.get('cited_by_count')}")
    print(f"  T: {w.get('title')}")
    print(f"  A: {(a[:850] if a else '(无摘要)')}")
    time.sleep(0.5)
