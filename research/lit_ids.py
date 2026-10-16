import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
def oa_src(name):
    r=requests.get("https://api.openalex.org/sources",headers=H,timeout=30,params={"search":name,"per-page":5})
    for s in r.json()["results"]:
        print(f"  {s['id'].split('/')[-1]:12s} | {s['display_name'][:60]:60s} | works={s['works_count']}")
for n in ["International Journal of Heat and Mass Transfer",
          "Applied Thermal Engineering",
          "International Journal of Thermal Sciences",
          "Journal of Quantitative Spectroscopy and Radiative Transfer",
          "International Communications in Heat and Mass Transfer",
          "Numerical Heat Transfer Part A"]:
    print(n); oa_src(n); time.sleep(0.4)
