# -*- coding: utf-8 -*-
import json, os, sys, time, urllib.parse, urllib.request
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
def s2(query, year="2000-2026", tries=5):
    u=("https://api.semanticscholar.org/graph/v1/paper/search/bulk?query="+urllib.parse.quote(query)
       +f"&fields=title,year,venue&year={year}")
    for a in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"s/1.0"}),timeout=45))
        except Exception as e:
            if a==tries-1: return {"error":str(e)}
            time.sleep(6*(a+1))
Q=[
 ("htd_total",      '"heat transfer deterioration"'),
 ("sco2_ht_total",  '"supercritical" "heat transfer deterioration"'),
 ("sco2_nn_total",  '"supercritical" "heat transfer" "neural network"'),
 ("sco2_op",        '"supercritical" "operator learning" OR "supercritical" "Fourier neural operator"'),
 ("sco2_ml_deter",  '"supercritical CO2" "machine learning"'),
 ("pseudo_crit",    '"pseudocritical" "machine learning"'),
 ("bouyancy_boehm", '"Dittus-Boelter" "correction"'),
 ("pcm_surr",       '"phase change material" "surrogate model"'),
 ("stefan_pinn",    '"Stefan problem" "physics-informed"'),
 ("nob_ml",         '"non-Oberbeck-Boussinesq"'),
 ("nob_ml2",        '"non-Oberbeck-Boussinesq" "machine learning"'),
 ("rb_surrogate",   '"Rayleigh-Benard" "surrogate model"'),
 ("fno_ht",         '"Fourier neural operator" "heat transfer"'),
 ("extrap_bench",   '"extrapolation" "surrogate model" "heat transfer"'),
]
for name, term in Q:
    d=s2(term)
    if "error" in d: print("[ERR ] %-14s %s"%(name,term)); time.sleep(4); continue
    print("[%5s] %-14s %s"%(d.get("total",0), name, term))
    for p in (d.get("data") or [])[:4]:
        print("        - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
