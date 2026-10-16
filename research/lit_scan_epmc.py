import requests, json, time
H={"User-Agent":"lit-scan/1.0"}
def epmc(q,n=25):
    for a in range(3):
        try:
            r=requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",headers=H,timeout=30,
              params={"query":q,"format":"json","pageSize":n,"resultType":"lite"})
            if r.status_code==200: return r.json()
        except Exception: pass
        time.sleep(2)
    return None
QS=['(TITLE_ABS:"surface radiation" OR TITLE_ABS:"thermal radiation") AND TITLE_ABS:"natural convection" AND (TITLE_ABS:surrogate OR TITLE_ABS:"machine learning" OR TITLE_ABS:neural OR TITLE_ABS:"reduced-order" OR TITLE_ABS:"multi-fidelity")',
 '(TITLE_ABS:radiation) AND TITLE_ABS:"natural convection" AND (TITLE_ABS:surrogate OR TITLE_ABS:"machine learning" OR TITLE_ABS:"neural network" OR TITLE_ABS:"reduced order model")',
 'TITLE_ABS:"Stark number" AND TITLE_ABS:"natural convection"',
 'TITLE_ABS:"Planck number" AND TITLE_ABS:"natural convection"',
 'TITLE_ABS:"conduction-radiation" AND TITLE_ABS:"natural convection"',
 'TITLE_ABS:"multi-fidelity" AND (TITLE_ABS:radiation OR TITLE_ABS:"heat transfer")',
 'TITLE_ABS:radiation AND TITLE_ABS:"natural convection" AND TITLE_ABS:review']
out={}
for q in QS:
    d=epmc(q)
    c=d["hitCount"] if d else None
    print(f"EPMC hit={c} | {q[:100]}")
    out[q]=d
    time.sleep(0.6)
json.dump(out,open("lit/scan_epmc.json","w",encoding="utf-8"),ensure_ascii=False)
print("saved")
