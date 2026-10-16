# -*- coding: utf-8 -*-
"""新型机械方向 x ML 占位"""
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
 # ---- 新概念结构 ----
 ("meta_total",     '"mechanical metamaterial"'),
 ("meta_no",        '"metamaterial" "neural operator"'),
 ("meta_fno",       '"metamaterial" "Fourier neural operator"'),
 ("meta_homog_ml",  '"metamaterial" "homogenization" "machine learning"'),
 ("auxetic_ml",     '"auxetic" "machine learning"'),
 ("origami_ml",     '"origami" "machine learning"'),
 ("kirigami_ml",    '"kirigami" "machine learning"'),
 ("tensegrity",     '"tensegrity"'),
 ("compliant_ml",   '"compliant mechanism" "machine learning"'),
 ("deployable_ml",  '"deployable structure" "machine learning"'),
 ("lattice_ml",     '"lattice structure" "machine learning"'),
 ("lattice_no",     '"lattice structure" "neural operator"'),
 # ---- 智能材料/驱动 ----
 ("sma_ml",         '"shape memory alloy" "machine learning"'),
 ("de_ml",          '"dielectric elastomer" "machine learning"'),
 ("mr_ml",          '"magnetorheological" "machine learning"'),
 ("softact_ml",     '"soft actuator" "machine learning"'),
 ("4dprint_ml",     '"4D printing" "machine learning"'),
 # ---- 新型传热结构（跟你资产衔接） ----
 ("tpms_total",     '"triply periodic minimal surface"'),
 ("tpms_ht",        '"triply periodic minimal surface" heat transfer'),
 ("tpms_ml",        '"triply periodic minimal surface" "machine learning"'),
 ("lattic_hex_ml",  '"lattice" "heat exchanger" "machine learning"'),
 ("hex_no",         '"heat exchanger" "neural operator"'),
 ("hex_ml",         '"heat exchanger" "machine learning"'),
 ("heatsink_ml",    '"heat sink" "topology optimization" "machine learning"'),
 ("topopt_ht",      '"topology optimization" "heat transfer"'),
 ("topopt_ht_ml",   '"topology optimization" "heat transfer" "machine learning"'),
 # ---- 新型能源机械 ----
 ("flywheel_ml",    '"flywheel energy storage" "machine learning"'),
 ("caes_ml",        '"compressed air energy storage" "machine learning"'),
 ("electrolyzer_ml",'"electrolyzer" "machine learning"'),
 ("pemfc_ml",       '"proton exchange membrane fuel cell" "machine learning"'),
 # ---- 微纳 ----
 ("mems_ml",        '"MEMS" "machine learning"'),
 ("nems_ml",        '"NEMS" OR "nanoelectromechanical" "machine learning"'),
 ("optomech",       '"optomechanics" "machine learning"'),
 # ---- 仿生 ----
 ("bionic_ml",      '"bionic robot" "machine learning"'),
 ("bioinsp_ml",     '"bio-inspired" "machine learning" design'),
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
json.dump(res, open(os.path.join(OUT,"s2_newmech2.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
