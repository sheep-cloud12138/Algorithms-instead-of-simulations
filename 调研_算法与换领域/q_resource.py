# -*- coding: utf-8 -*-
"""算法创新的资源约束：算力到底是不是瓶颈"""
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
 # ---- 小资源能不能出成果 ----
 ("small_data_no",  '"neural operator" "small data" OR "neural operator" "limited data"'),
 ("few_shot_pde",   '"few-shot" "partial differential equation" OR "few-shot" "neural operator"'),
 ("efficient_no",   '"efficient" "neural operator" "training cost" OR "lightweight" "neural operator"'),
 ("simple_better",  '"simple baseline" outperform deep learning'),
 ("neg_result",     '"negative result" machine learning physics OR "negative results" deep learning science'),
 ("reproduce_crisis",'"reproducibility crisis" deep learning'),
 ("no_pretrain",    '"without pretraining" "neural operator" OR "training from scratch" PDE surrogate'),
 # ---- 什么规模才需要大算力 ----
 ("fm_compute",     '"foundation model" "PDE" "pretraining" compute'),
 ("large_pretrain", '"large-scale" "pretraining" "partial differential equations"'),
 ("dpot",           '"PDE pretraining" OR "PDE pre-training"'),
 ("scaling_llm",    '"scaling law" "pretraining" compute optimal'),
 # ---- 对照：基线复现成本 ----
 ("baseline_study", '"benchmark study" "neural operator" OR "empirical study" surrogate modeling PDE'),
 ("ablation",       '"ablation study" "neural operator"'),
 ("hyperparam",     '"hyperparameter" "neural operator" sensitivity'),
 # ---- 单人/小组产出 ----
 ("single_author",  '"Fourier neural operator" '),
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
json.dump(res, open(os.path.join(OUT,"s2_resource.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
