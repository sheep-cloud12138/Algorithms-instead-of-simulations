# -*- coding: utf-8 -*-
"""纯算法创新：理论侧 / 方法期刊 vs 顶会 / 可行切口"""
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
 # ---- 理论侧（可能不需要 GPU） ----
 ("th_approx",     '"neural operator" approximation theory OR "neural operator" approximation rate'),
 ("th_gen",        '"generalization bound" "neural operator" OR "generalization bound" "operator learning"'),
 ("th_disc",       '"discretization invariance" OR "discretization-invariant" neural operator'),
 ("th_physics",    '"physics-informed" "convergence" "neural network" analysis'),
 ("th_scale",      '"scaling law" "neural operator" OR "scaling laws" "scientific machine learning"'),
 # ---- 方法与可信度 ----
 ("uq_no",         '"uncertainty quantification" "neural operator"'),
 ("trust_region",  '"trustworthy" OR "reliable" "neural operator"'),
 ("ood_bound",     '"out-of-distribution" "neural operator" bound'),
 ("certified",     '"certified" OR "verified" neural network PDE'),
 ("active_theory", '"active learning" "surrogate" "theoretical" OR "active learning" "sample complexity"'),
 # ---- 基准 / 数据集（方法性产物） ----
 ("bench_sciml",   '"benchmark" "scientific machine learning"'),
 ("bench_pde",     '"PDE" benchmark "neural operator"'),
 ("bench_fair",    '"fair comparison" OR "fair benchmark" neural operator'),
 ("reproduc",      '"reproducibility" "scientific machine learning" OR "reproducibility" "neural operator"'),
 # ---- 顶会生态 ----
 ("neurips_sciml", '"neural operator" NeurIPS OR ICML OR ICLR'),
 ("foundation_pde",'"PDE foundation model" OR "foundation model" "neural operator"'),
 ("llm_sciml",     '"large language model" "scientific machine learning"'),
 ("agent_sciml",   '"LLM agent" simulation OR "agent" "scientific discovery"'),
 # ---- 对照：老牌算法红海 ----
 ("kan_total",     '"Kolmogorov-Arnold network"'),
 ("diffusion_sci", '"diffusion model" "scientific machine learning" OR "diffusion model" PDE'),
 ("gnn_sci",       '"graph neural network" "partial differential equation"'),
 ("mamba_sci",     '"Mamba" "partial differential equation" OR "state space model" PDE'),
]
res={}
for name, term in Q:
    d=s2(term)
    if "error" in d:
        print("[ERR ] %-15s %s"%(name,term)); res[name]={"err":d["error"]}; time.sleep(4); continue
    tot=d.get("total",0); data=d.get("data") or []
    res[name]={"q":term,"total":tot,"top":[{"y":p.get("year"),"t":p.get("title"),"v":p.get("venue")} for p in data[:6]]}
    print("[%7s] %-15s %s"%(tot,name,term))
    for p in data[:3]: print("          - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
json.dump(res, open(os.path.join(OUT,"s2_purealgo.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
