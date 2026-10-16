import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
QS=['all:"surface radiation" AND all:"natural convection" AND (all:surrogate OR all:"machine learning" OR all:"neural network" OR all:"reduced-order")',
 'all:"radiation" AND all:"natural convection" AND all:"surrogate model"',
 'all:"radiative transfer" AND all:"natural convection" AND all:"neural"',
 'all:"radiation" AND all:"natural convection" AND all:"multi-fidelity"',
 'all:"similarity" AND all:"radiation" AND all:"natural convection"']
out={}
for q in QS:
    d=None
    for a in range(5):
        try:
            r=requests.get("http://export.arxiv.org/api/query",headers=H,timeout=30,
              params={"search_query":q,"max_results":25,"sortBy":"relevance"})
            if r.status_code==200: d=r.text; break
            time.sleep(5+a*5)
        except Exception: time.sleep(4)
    if d is None:
        print(f"arXiv FAIL | {q}"); out[q]="FAIL"; continue
    import re
    ents=re.findall(r"<entry>(.*?)</entry>", d, re.S)
    print(f"\n### arXiv n={len(ents)} | {q[:80]}")
    for e in ents[:14]:
        t=re.search(r"<title>(.*?)</title>",e,re.S).group(1).strip().replace("\n"," ")
        idu=re.search(r"<id>(.*?)</id>",e,re.S).group(1).strip()
        pub=re.search(r"<published>(.*?)</published>",e,re.S).group(1)[:7]
        print(f"  {pub} | {t[:88]} | {idu}")
    out[q]=d
    time.sleep(4)
json.dump(out,open("lit/scan_arxiv.json","w",encoding="utf-8"),ensure_ascii=False)
print("\nsaved")
