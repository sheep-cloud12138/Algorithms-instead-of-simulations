# -*- coding: utf-8 -*-
import json, os, sys, time, urllib.parse, urllib.request
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
def s2(q, year="2000-2026", tries=6):
    u=("https://api.semanticscholar.org/graph/v1/paper/search/bulk?query="+urllib.parse.quote(q)
       +f"&fields=title,year,venue&year={year}")
    for a in range(tries):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"s/1.0"}),timeout=45))
        except Exception as e:
            if a==tries-1: return {"error":str(e)}
            time.sleep(7*(a+1))
Q=[
 ("no_elasticity",   '"neural operator" "elasticity"'),
 ("no_solidmech",    '"neural operator" "solid mechanics"'),
 ("no_plasticity",   '"neural operator" "plasticity"'),
 ("no_fatigue",      '"neural operator" "fatigue"'),
 ("no_fracture",     '"neural operator" "fracture"'),
 ("no_contact",      '"neural operator" "contact mechanics"'),
 ("fno_solid",       '"Fourier neural operator" "solid mechanics"'),
 ("deeponet_solid",  '"DeepONet" "elasticity" OR "DeepONet" "solid"'),
 ("no_multiscale",   '"neural operator" "multiscale" materials'),
 ("no_micro",        '"neural operator" "microstructure"'),
 ("no_gearbox",      '"neural operator" "gear"'),
 ("no_machinetool",  '"neural operator" "machine tool"'),
 ("no_therm_err",    '"neural operator" "thermal error"'),
 ("no_vehicle",      '"neural operator" "vehicle"'),
 ("no_battery",      '"neural operator" "battery"'),
]
res={}
for name, term in Q:
    d=s2(term)
    if "error" in d:
        print("[ERR ] %-16s %s"%(name,term)); res[name]={"err":d["error"]}; time.sleep(4); continue
    tot=d.get("total",0); data=d.get("data") or []
    res[name]={"q":term,"total":tot,"top":[{"y":p.get("year"),"t":p.get("title"),"v":p.get("venue")} for p in data[:6]]}
    print("[%5s] %-16s %s"%(tot,name,term))
    for p in data[:3]: print("         - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
json.dump(res, open(os.path.join(OUT,"s2_solid_op.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
