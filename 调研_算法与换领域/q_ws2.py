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
 ("ws_ml",           '"winding switching" "machine learning"'),
 ("ws_ml2",          '"winding switching" neural network'),
 ("ws_opt",          '"winding switching" optimization'),
 ("ws_reconfig2",    '"winding reconfiguration"'),
 ("ws_reconfig_ml",  '"winding reconfiguration" "machine learning"'),
 ("ws_reconfig_ctrl",'"winding reconfiguration" control'),
 ("ws_switch_trans", '"switching transient" motor winding'),
 ("ws_splitphase",   '"split-phase" motor OR "split winding" machine'),
 ("ws_openwind",     '"open-end winding" motor'),
 ("ws_multimode",    '"multi-mode" motor winding'),
 ("ws_mode_trans",   '"mode transition" motor drive'),
 ("mec_total",       '"magnetic equivalent circuit"'),
 ("mec_ml",          '"magnetic equivalent circuit" "machine learning"'),
 ("mec_nn",          '"magnetic equivalent circuit" neural network'),
 ("dq_ml",           '"dq" model "machine learning" motor'),
 ("reluctance_ml",   '"switched reluctance" "machine learning"'),
 ("ws_doe",          '"winding switching" design of experiments'),
]
for name, term in Q:
    d=s2(term)
    if "error" in d:
        print("[ERR ] %-18s %s"%(name,term)); time.sleep(4); continue
    tot=d.get("total",0); data=d.get("data") or []
    print("[%7s] %-18s %s"%(tot,name,term))
    for p in data[:4]: print("          - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
