import requests, time
H={"User-Agent":"lit-scan/1.0"}
for d in ["10.1063/5.0315076","10.1007/s13201-025-02591-2","10.1177/03019233241276289"]:
    for a in range(4):
        r=requests.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{d}",headers=H,timeout=40,
          params={"fields":"title,year,venue,abstract"})
        if r.status_code==200: break
        if r.status_code==429: time.sleep(6+a*6); continue
        break
    if r.status_code!=200: print(f"\n### {d} ERR {r.status_code}"); time.sleep(2); continue
    p=r.json()
    print(f"\n### {d}\n  {p.get('year')} | {p.get('venue')}\n  T: {p.get('title')}\n  A: {(p.get('abstract') or '(无摘要)')[:950]}")
    time.sleep(3.5)
