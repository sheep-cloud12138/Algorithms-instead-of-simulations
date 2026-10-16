import requests, re, time, json
H={"User-Agent":"lit-scan/1.0"}
QS=['all:"surface radiation" AND all:"natural convection" AND (all:surrogate OR all:"machine learning" OR all:"neural network" OR all:"reduced-order")',
 'all:"radiation" AND all:"natural convection" AND all:"surrogate"',
 'all:"radiative" AND all:"Rayleigh-Benard" AND all:"reduced-order"',
 'all:"radiation" AND all:"natural convection" AND all:"multi-fidelity"',
 'all:"similarity" AND all:"radiation" AND all:"natural convection" AND all:"scale analysis"']
out={}
for q in QS:
    d=None
    for a in range(4):
        try:
            r=requests.get("http://export.arxiv.org/api/query",headers=H,timeout=40,
              params={"search_query":q,"max_results":20,"sortBy":"relevance"})
            if r.status_code==200: d=r.text; break
            time.sleep(6+a*6)
        except Exception: time.sleep(5)
    if d is None: print(f"arXiv FAIL | {q[:70]}"); out[q]="FAIL"; continue
    ents=re.findall(r"<entry>(.*?)</entry>",d,re.S)
    print(f"\n### arXiv n={len(ents)} | {q[:78]}")
    for e in ents[:12]:
        t=re.search(r"<title>(.*?)</title>",e,re.S).group(1).strip().replace("\n"," ")
        idu=re.search(r"<id>(.*?)</id>",e,re.S).group(1).strip()
        pub=re.search(r"<published>(.*?)</published>",e,re.S).group(1)[:7]
        print(f"  {pub} | {t[:86]} | {idu}")
    out[q]=[re.search(r"<title>(.*?)</title>",e,re.S).group(1).strip().replace("\n"," ") for e in ents]
    time.sleep(5)
json.dump(out,open("lit/scan_arxiv.json","w",encoding="utf-8"),ensure_ascii=False)
print("saved")
