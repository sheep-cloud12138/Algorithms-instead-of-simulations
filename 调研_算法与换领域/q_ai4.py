# -*- coding: utf-8 -*-
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
 ("ai_mech_review",   '"artificial intelligence" "mechanical engineering" review'),
 ("ai_veh_review",    '"artificial intelligence" "vehicle engineering" OR "automotive" review'),
 ("phm_dl",           '"fault diagnosis" "deep learning"'),
 ("phm_bearing_ds",   '"bearing fault diagnosis" "deep learning"'),
 ("batt_soh_ml",      '"state of health" "lithium-ion battery" "machine learning"'),
 ("batt_soc_ml",      '"state of charge" estimation "deep learning"'),
 ("topopt_dl",        '"topology optimization" "deep learning"'),
 ("gen_design",       '"generative design" "deep learning" mechanical'),
 ("robot_ml",         '"robot" "reinforcement learning" control'),
 ("digi_twin_mech",   '"digital twin" "machine tool" OR "digital twin" manufacturing'),
 ("additive_ml",      '"additive manufacturing" "machine learning" review'),
 ("autodrive_ml",     '"autonomous driving" "deep learning"'),
 ("veh_dyn_ml",       '"vehicle dynamics" "machine learning"'),
 ("mech_fatigue_ds",  '"fatigue" "deep learning" life prediction review'),
 ("surrogate_cae",    '"surrogate model" "finite element" "machine learning" design optimization'),
 ("mech_thermal_err", '"thermal error" "machine tool"'),
 ("gpu_feasible",     '"Fourier neural operator" mechanics OR structural OR elasticity'),
 ("fno_solid",        '"neural operator" "elasticity" OR "solid mechanics"'),
 ("fno_tribo",        '"neural operator" "contact" OR "tribology"'),
 ("mech_archard_fem", '"wear simulation" "finite element" "machine learning"'),
]
res={}
for name, term in Q:
    d=s2(term)
    if "error" in d:
        print("[ERR ] %-18s %s"%(name,term)); res[name]={"err":d["error"]}; time.sleep(4); continue
    tot=d.get("total",0); data=d.get("data") or []
    res[name]={"q":term,"total":tot,"top":[{"y":p.get("year"),"t":p.get("title"),"v":p.get("venue")} for p in data[:6]]}
    print("[%6s] %-18s %s"%(tot,name,term))
    for p in data[:3]: print("         - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
json.dump(res, open(os.path.join(OUT,"s2_ai4me_veh.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
