# -*- coding: utf-8 -*-
"""绕组切换 + AI4电机控制（可仿真类）"""
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
 # ===== 绕组切换 =====
 ("ws_total",       '"winding switching"'),
 ("ws_reconfig",    '"winding reconfiguration" OR "winding reconfigurable"'),
 ("ws_motor",       '"winding switching" motor'),
 ("ws_traction",    '"winding switching" traction OR "winding changeover" traction'),
 ("ws_stardelta",   '"star-delta" switching motor'),
 ("ws_seriespar",   '"series-parallel" winding switching OR "series/parallel" winding motor'),
 ("ws_polechange",  '"pole changing" motor OR "pole-changing"'),
 ("ws_variableflux",'"variable flux" motor OR "variable-flux" machine'),
 ("ws_widespeed",   '"wide speed range" motor winding'),
 ("ws_transient",   '"winding switching" transient OR "winding switching" torque ripple'),
 ("ws_control",     '"winding switching" control strategy'),
 ("ws_ev",          '"winding switching" electric vehicle OR "winding reconfiguration" EV'),
 ("ws_ml",          '"winding switching" "machine learning" OR "winding reconfiguration" optimization'),
 ("ws_opt",         '"winding switching" optimization OR "winding reconfiguration" optimization'),
 # ===== 低 fidelity 底座（磁路法） =====
 ("mec_total",      '"magnetic equivalent circuit"'),
 ("mec_ml",         '"magnetic equivalent circuit" "machine learning" OR "magnetic equivalent circuit" neural'),
 ("dq_ml",          '"dq" model "machine learning" motor OR "dq-axis" neural network'),
 ("lptn_ignore",    '"magnetic equivalent circuit" electric machine'),
 # ===== AI + 电机控制（可仿真） =====
 ("mpc_pmsm",       '"model predictive control" PMSM'),
 ("mpc_pmsm_ml",    '"model predictive control" PMSM "machine learning" OR "model predictive control" PMSM neural'),
 ("sensorless_ml2", '"sensorless" PMSM "machine learning" OR "sensorless" PMSM neural network'),
 ("param_id_ml",    '"parameter identification" motor "machine learning" OR "parameter estimation" PMSM neural'),
 ("torque_ripple_ml",'"torque ripple" "machine learning" motor OR "torque ripple" neural network PMSM'),
 ("flt_tol_ml",     '"fault-tolerant" motor drive "machine learning" OR "fault tolerant" PMSM neural'),
 ("current_ctrl_rl",'"current control" "reinforcement learning" motor'),
 ("drive_surr",     '"motor drive" surrogate model'),
 ("em_sim_ml",      '"finite element" motor "machine learning" surrogate'),
 ("em_digitaltwin", '"electric machine" "digital twin"'),
]
for name, term in Q:
    d=s2(term)
    if "error" in d:
        print("[ERR ] %-18s %s"%(name,term)); time.sleep(4); continue
    tot=d.get("total",0); data=d.get("data") or []
    print("[%7s] %-18s %s"%(tot,name,term))
    for p in data[:3]: print("          - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
