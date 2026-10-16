import requests, json, time
H={"User-Agent":"lit-scan/1.0"}
DOIS=["10.1016/j.ijheatmasstransfer.2026.129230","10.1016/j.icheatmasstransfer.2025.109805",
 "10.1016/j.icheatmasstransfer.2026.111511","10.1016/j.ijheatmasstransfer.2014.02.009",
 "10.1016/j.rineng.2024.102514","10.1016/J.ENCONMAN.2013.01.024",
 "10.1016/J.IJHEATMASSTRANSFER.2016.11.061","10.1016/j.fusengdes.2026.115789",
 "10.1016/j.ijthermalsci.2020.106310","10.1016/J.IJHEATMASSTRANSFER.2006.07.022"]
for d in DOIS:
    for a in range(4):
        r=requests.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{d}",headers=H,timeout=40,
          params={"fields":"title,year,venue,abstract,externalIds,tldr"})
        if r.status_code==200: break
        if r.status_code==429: time.sleep(6+a*6); continue
        break
    if r.status_code!=200: print(f"\n### {d} ERR {r.status_code}"); time.sleep(2); continue
    p=r.json()
    print(f"\n### {d}\n  {p.get('year')} | {p.get('venue')}\n  T: {p.get('title')}")
    print(f"  A: {(p.get('abstract') or '(无摘要)')[:900]}")
    time.sleep(3.5)
