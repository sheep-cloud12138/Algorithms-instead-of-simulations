import json
d=json.load(open("lit/scan_epmc.json",encoding="utf-8"))
for q,v in d.items():
    if not v: print("ERR:",q[:70]); continue
    if v["hitCount"]==0: continue
    print(f"\n=== {q[:95]} (hit={v['hitCount']}) ===")
    for r in v["resultList"]["result"]:
        print(f"  {r.get('pubYear')} | {str(r.get('journalTitle'))[:38]:38s} | {str(r.get('title'))[:90]}")
        print(f"        DOI:{r.get('doi')} src={r.get('source')} id={r.get('id')}")
