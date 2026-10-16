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
 # ---- 新兴机械 ----
 ("embodied_ai",    '"embodied intelligence" OR "embodied AI"'),
 ("humanoid",       '"humanoid robot"'),
 ("humanoid_learn", '"humanoid robot" "reinforcement learning" OR "humanoid robot" "learning"'),
 ("softrobot_ml",   '"soft robot" "machine learning"'),
 ("softrobot_ctrl", '"soft robot" control'),
 ("metamaterial_ml",'"mechanical metamaterial" "machine learning"'),
 ("metamaterial_dl",'"mechanical metamaterial" "deep learning"'),
 ("tactile_ml",     '"tactile sensing" "machine learning"'),
 ("wearable_ml",    '"wearable" "machine learning" sensor'),
 ("microrobot",     '"microrobot" OR "micro/nano robot"'),
 ("selfhealing",    '"self-healing" material machine learning'),
 ("digitaltwin_ml", '"digital twin" "machine learning" manufacturing'),
 ("aiforscience_mech",'"AI for science" mechanics OR "AI for science" mechanical'),
 ("foundation_mech",'"foundation model" "mechanical" OR "foundation model" "engineering design"'),
 ("llm_design",     '"large language model" "engineering design" OR "large language model" CAD'),
 ("llm_robot",      '"large language model" "robotics" OR "vision language action"'),
 # ---- 传统机械（做对照） ----
 ("tribo_total",    '"tribology"'),
 ("tribo_ml",       '"tribology" "machine learning"'),
 ("machinetool_tot",'"machine tool"'),
 ("fatigue_total",  '"fatigue" life prediction'),
 ("topopt_total",   '"topology optimization"'),
 ("additive_total", '"additive manufacturing"'),
 ("intelligent_mfg",'"intelligent manufacturing" OR "smart manufacturing"'),
 ("phm_total",      '"prognostics and health management"'),
]
res={}
for name, term in Q:
    d=s2(term)
    if "error" in d:
        print("[ERR ] %-18s %s"%(name,term)); res[name]={"err":d["error"]}; time.sleep(4); continue
    tot=d.get("total",0); data=d.get("data") or []
    res[name]={"q":term,"total":tot,"top":[{"y":p.get("year"),"t":p.get("title"),"v":p.get("venue")} for p in data[:6]]}
    print("[%7s] %-18s %s"%(tot,name,term))
    for p in data[:3]: print("          - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
json.dump(res, open(os.path.join(OUT,"s2_newmech.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
