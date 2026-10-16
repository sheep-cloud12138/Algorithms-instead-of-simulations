# -*- coding: utf-8 -*-
"""低算力方法方向：占位与可行性"""
import json, os, sys, time, urllib.parse, urllib.request
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
def s2(q, year="2005-2026", tries=6):
    u=("https://api.semanticscholar.org/graph/v1/paper/search/bulk?query="+urllib.parse.quote(q)
       +f"&fields=title,year,venue&year={year}")
    for a in range(tries):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"s/1.0"}),timeout=45))
        except Exception as e:
            if a==tries-1: return {"error":str(e)}
            time.sleep(7*(a+1))
Q=[
 ("conformal_no",   '"conformal prediction" "neural operator" OR "conformal prediction" surrogate model'),
 ("conformal_pde",  '"conformal prediction" "partial differential equation"'),
 ("appl_domain",    '"applicability domain" "machine learning"'),
 ("appl_domain_sim",'"applicability domain" surrogate OR "applicability domain" "simulation"'),
 ("aposteriori",    '"a posteriori error estimation" "machine learning" OR "a posteriori" surrogate error'),
 ("ood_detect",     '"out-of-distribution detection" "surrogate" OR "out-of-distribution detection" "scientific"'),
 ("rollout",        '"long-term" stability "neural operator" OR "rollout" error "neural operator"'),
 ("spectral_bias",  '"spectral bias" "neural network" OR "spectral bias" "PDE"'),
 ("sample_eff",     '"sample efficiency" "neural operator" OR "data efficiency" "neural operator"'),
 ("active_no",      '"active learning" "neural operator"'),
 ("data_eff_surr",  '"data efficiency" "surrogate model"'),
 ("error_est_no",   '"error estimate" "neural operator" OR "error bound" "neural operator" computed'),
 ("physics_consist",'"physical consistency" "neural operator" OR "conservation" "neural operator" violation'),
 ("equi_no",        '"equivariant" "neural operator"'),
 ("symmetry_break", '"symmetry breaking" "machine learning" OR "symmetry" "neural network" physics conservation'),
]
res={}
for name, term in Q:
    d=s2(term)
    if "error" in d:
        print("[ERR ] %-16s %s"%(name,term)); res[name]={"err":d["error"]}; time.sleep(4); continue
    tot=d.get("total",0); data=d.get("data") or []
    res[name]={"q":term,"total":tot,"top":[{"y":p.get("year"),"t":p.get("title"),"v":p.get("venue")} for p in data[:6]]}
    print("[%7s] %-16s %s"%(tot,name,term))
    for p in data[:3]: print("          - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
json.dump(res, open(os.path.join(OUT,"s2_lightalgo.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
