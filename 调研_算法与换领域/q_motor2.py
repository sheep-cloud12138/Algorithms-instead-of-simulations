# -*- coding: utf-8 -*-
import json, os, sys, time, urllib.parse, urllib.request
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
def s2(q, year="2005-2026", tries=6):
    u=("https://api.semanticscholar.org/graph/v1/paper/search/bulk?query="+urllib.parse.quote(q)
       +f"&fields=title,year,venue&year={year}")
    for a in range(tries):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"s/1.0"}),timeout=45))
        except Exception as e:
            if a==tries-1: return {"error":str(e)}
            time.sleep(7*(a+1))
Q=[
 ("steinmetz_ml",   '"Steinmetz" "machine learning"'),
 ("steinmetz_nn",   '"Steinmetz" "neural network"'),
 ("steinmetz_eq",   '"Steinmetz equation"'),
 ("ironloss_ml",    '"iron loss" "machine learning"'),
 ("coreloss_ml",    '"core loss" "machine learning"'),
 ("ironloss_model", '"iron loss model"'),
 ("coreloss_model", '"core loss model"'),
 ("coffinmanson",   '"Coffin-Manson"'),
 ("coffin_ml",      '"Coffin-Manson" "machine learning"'),
 ("coffin_nn",      '"Coffin-Manson" "neural network"'),
 ("thermcycle_life",'"thermal cycling" "remaining useful life" power module'),
 ("arrhenius_ml",   '"Arrhenius" "machine learning" lifetime'),
 ("igbt_ml",        '"IGBT" "machine learning"'),
 ("igbt_rul",       '"IGBT" "remaining useful life"'),
 ("powermodule_rul",'"power module" "remaining useful life"'),
 ("elec_themal_ml", '"electric machine" "thermal" model machine learning'),
 ("lossmap_ml",     '"loss map" OR "efficiency map" "machine learning" motor'),
 ("mag_no",         '"neural operator" Maxwell equations'),
]
for name, term in Q:
    d=s2(term)
    if "error" in d:
        print("[ERR ] %-18s %s"%(name,term)); time.sleep(4); continue
    tot=d.get("total",0); data=d.get("data") or []
    print("[%7s] %-18s %s"%(tot,name,term))
    for p in data[:3]: print("          - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
