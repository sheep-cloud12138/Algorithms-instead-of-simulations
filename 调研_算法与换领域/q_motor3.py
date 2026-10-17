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
 ("lptn_total",     '"lumped parameter thermal network"'),
 ("lptn_motor",     '"lumped parameter thermal network" motor OR "lumped parameter thermal network" machine'),
 ("lptn_ml",        '"thermal network" "machine learning" electric machine OR "thermal network" "neural network" motor'),
 ("winding_temp_ml",'"winding temperature" "machine learning" OR "winding temperature" "neural network" motor'),
 ("motor_temp_est", '"rotor temperature" estimation "machine learning" OR "magnet temperature" estimation neural'),
 ("traction_motor_th",'"traction motor" thermal'),
 ("motor_th_surr",  '"electric motor" thermal surrogate OR "electric machine" thermal surrogate'),
 ("motor_cfd",      '"electric machine" OR "electric motor" CFD thermal'),
 ("motor_thermal_net",'"thermal model" "electric machine" neural network'),
 ("em_thermal_dt",  '"electric machine" "digital twin" thermal'),
 ("motor_loss_heat",'"loss distribution" "electric machine" thermal'),
 ("pm_temp_ml",     '"magnet temperature" "machine learning"'),
 ("motor_th_review",'"thermal management" "electric machine" review'),
 ("igbt_thermal_ml",'"IGBT" OR "power module" "thermal" "machine learning"'),
 ("electrothermal_ml",'"electro-thermal" "machine learning" power module OR "electrothermal" surrogate'),
 ("pm_package_life",'"solder" "Coffin-Manson" OR "solder joint" fatigue life machine learning'),
 ("sic_thermal_ml", '"SiC" "power module" "machine learning" thermal'),
]
for name, term in Q:
    d=s2(term)
    if "error" in d:
        print("[ERR ] %-18s %s"%(name,term)); time.sleep(4); continue
    tot=d.get("total",0); data=d.get("data") or []
    print("[%7s] %-18s %s"%(tot,name,term))
    for p in data[:3]: print("          - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
