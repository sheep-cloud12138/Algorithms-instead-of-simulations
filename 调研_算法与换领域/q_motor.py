# -*- coding: utf-8 -*-
"""AI4电机 / AI4电控 的占位与相似律破缺"""
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
 # ===== 电机本体 =====
 ("em_total",      '"electric machine" "machine learning"'),
 ("em_surr",       '"electric machine" surrogate'),
 ("motor_design_ml",'"electric motor" "machine learning" design'),
 ("motor_topopt",  '"electric machine" OR "electric motor" topology optimization'),
 ("em_no",         '"neural operator" "electromagnetic" OR "neural operator" magnetics'),
 ("em_fea_ml",     '"electromagnetic" "finite element" surrogate "machine learning"'),
 ("pm_demag_ml",   '"demagnetization" "machine learning" permanent magnet'),
 ("motor_fault",   '"motor fault diagnosis" "deep learning"'),
 ("motor_phm",     '"electric machine" "prognostics" OR "electric machine" "condition monitoring" "deep learning"'),
 ("motor_thermal", '"electric motor" "thermal" "machine learning" OR "electric machine" thermal model neural'),
 ("motor_sensorless",'"sensorless control" "machine learning" OR "sensorless" neural network'),
 ("motor_scaling", '"scaling law" "electric machine" OR "scaling laws" motor'),
 # ===== 相似律（关键） =====
 ("steinmetz",     '"Steinmetz"'),
 ("steinmetz_ml",  '"Steinmetz" "machine learning" OR "Steinmetz" "neural network"'),
 ("ironloss_ml",   '"iron loss" "machine learning" OR "core loss" "machine learning"'),
 ("ironloss_model",'"iron loss model" OR "core loss model"'),
 ("coffinmanson",  '"Coffin-Manson"'),
 ("coffin_ml",     '"Coffin-Manson" "machine learning" OR "Coffin-Manson" "neural network"'),
 ("thermal_cycle_life",'"thermal cycling" lifetime "machine learning" power module'),
 ("arrhenius_ml",  '"Arrhenius" "machine learning" lifetime'),
 # ===== 电控 / 电力电子 =====
 ("igbt_ml",       '"IGBT" "machine learning"'),
 ("igbt_life",     '"IGBT" OR "power module" "remaining useful life"'),
 ("sic_ml",        '"silicon carbide" "machine learning" OR "SiC MOSFET" "machine learning"'),
 ("inverter_fault",'"inverter" "fault diagnosis" "deep learning"'),
 ("converter_ml",  '"power converter" "machine learning" control'),
 ("mpc_ml_drive",  '"model predictive control" motor drive "machine learning"'),
 ("rl_motor",      '"reinforcement learning" motor control'),
 ("rl_converter",  '"reinforcement learning" power electronics OR "reinforcement learning" converter'),
 ("pmsm_ml",       '"permanent magnet synchronous motor" "machine learning" OR "PMSM" "deep learning"'),
 ("ev_powertrain_ml",'"electric vehicle" powertrain "machine learning"'),
 ("bms_ml",        '"battery management system" "machine learning"'),
 ("emc_ml",        '"electromagnetic compatibility" machine learning OR "EMI" "machine learning"'),
]
res={}
for name, term in Q:
    d=s2(term)
    if "error" in d:
        print("[ERR ] %-18s %s"%(name,term)); res[name]={"err":d["error"]}; time.sleep(4); continue
    tot=d.get("total",0); data=d.get("data") or []
    res[name]={"q":term,"total":tot,"top":[{"y":p.get("year"),"t":p.get("title"),"v":p.get("venue")} for p in data[:6]]}
    print("[%7s] %-18s %s"%(tot,name,term))
    for p in data[:2]: print("          - (%s) %s | %s"%(p.get("year"),p.get("title"),p.get("venue")))
    time.sleep(4)
json.dump(res, open(os.path.join(OUT,"s2_motor.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
