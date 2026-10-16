# -*- coding: utf-8 -*-
"""生成式模型（flow matching / diffusion / consistency）在 SciML 的占位"""
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
 ("fm_total",       '"flow matching"'),
 ("fm_pde",         '"flow matching" "partial differential equation"'),
 ("fm_physics",     '"flow matching" physics OR "flow matching" "physical simulation"'),
 ("fm_surrogate",   '"flow matching" surrogate'),
 ("fm_inverse",     '"flow matching" "inverse problem"'),
 ("fm_uq",          '"flow matching" "uncertainty quantification"'),
 ("fm_operator",    '"flow matching" "neural operator" OR "flow matching" "operator learning"'),
 ("fm_field",       '"flow matching" "field" generation OR "flow matching" turbulence'),
 ("fm_cfd",         '"flow matching" "computational fluid dynamics" OR "flow matching" CFD'),
 ("fm_geo",         '"flow matching" geophysics OR "flow matching" climate OR "flow matching" weather'),
 ("fm_gen_eng",     '"flow matching" engineering design OR "flow matching" topology optimization'),
 ("rf_total",       '"rectified flow"'),
 ("rf_pde",         '"rectified flow" "partial differential equation" OR "rectified flow" physics'),
 ("si_total",       '"stochastic interpolants"'),
 ("consistency_pde",'"consistency model" "partial differential equation" OR "consistency model" PDE'),
 ("diff_pde",       '"diffusion model" "partial differential equation" surrogate'),
 ("diff_inverse",   '"diffusion model" "inverse problem" physics'),
 ("gen_ensemble",   '"generative model" ensemble "partial differential equation"'),
 ("onestep_pde",    '"one-step" OR "few-step" generative "partial differential equation"'),
 ("fm_distill",     '"flow matching" distillation OR "flow matching" "few-step sampling"'),
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
json.dump(res, open(os.path.join(OUT,"s2_flowmatch.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
