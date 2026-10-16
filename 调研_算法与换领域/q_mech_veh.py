# -*- coding: utf-8 -*-
"""机械 / 车辆 方向的占位检索 (Semantic Scholar)"""
import json, os, sys, time, urllib.parse, urllib.request
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

def s2(query, year="2005-2026", tries=5):
    u = ("https://api.semanticscholar.org/graph/v1/paper/search/bulk?query="
         + urllib.parse.quote(query) + f"&fields=title,year,venue&year={year}")
    for a in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent":"s/1.0"}), timeout=45))
        except Exception as e:
            if a == tries-1: return {"error": str(e)}
            time.sleep(6*(a+1))

Q = [
 # ---------- 车辆 ----------
 ("veh_batt_thermal",  '"battery thermal management" "machine learning"'),
 ("veh_batt_twin",     '"battery" "digital twin" "thermal"'),
 ("veh_crash_ml",      '"crashworthiness" "machine learning"'),
 ("veh_crash_surr",    '"crashworthiness" surrogate'),
 ("veh_crash_scale",   '"crashworthiness" "scaling law"'),
 ("veh_crash_scalemod",'"scale model" crashworthiness similarity'),
 ("veh_nvh_ml",        '"NVH" "machine learning"'),
 ("veh_nvh_surr",      '"vehicle interior noise" surrogate OR "vibration" "machine learning" surrogate'),
 ("veh_aero_ml",       '"vehicle aerodynamics" "machine learning"'),
 ("veh_brake_ml",      '"brake" "thermal" "machine learning"'),
 ("veh_tire_ml",       '"tire" "machine learning" surrogate'),
 ("veh_fatigue_ml",    '"fatigue life" "machine learning" vehicle'),
 ("veh_transmission_ml",'"transmission" "gear" "machine learning"'),
 # ---------- 机械 ----------
 ("mech_wear_ml",      '"wear" "machine learning" tribology'),
 ("mech_archard",      '"Archard"'),
 ("mech_archard_ml",   '"Archard" "machine learning"'),
 ("mech_gear_ml",      '"gear" "machine learning" surrogate'),
 ("mech_bearing_ml",   '"rolling bearing" "machine learning" surrogate'),
 ("mech_tool_error",   '"thermal error" "machine tool" "machine learning"'),
 ("mech_welding_ml",   '"welding" "machine learning" surrogate'),
 ("mech_additive_ml",  '"additive manufacturing" "machine learning" surrogate melt pool'),
 ("mech_rotor_ml",     '"rotor dynamics" "machine learning"'),
 ("mech_seal_ml",      '"mechanical seal" "machine learning"'),
 ("mech_creep_ml",     '"creep" "machine learning" life prediction'),
 ("mech_larson",       '"Larson-Miller"'),
 ("mech_larson_ml",    '"Larson-Miller" "machine learning"'),
 ("mech_fatigue_ml",   '"fatigue life prediction" "machine learning"'),
 ("mech_sn_curve",     '"S-N curve" "machine learning"'),
 ("mech_miner_ml",     '"Miner" "fatigue" "machine learning"'),
 ("mech_size_eff_fat", '"size effect" "fatigue" "machine learning"'),
 ("mech_spindle_ml",   '"spindle" "thermal" "machine learning"'),
 ("mech_friction_ml",  '"friction" "machine learning" tribology surrogate'),
 ("mech_contact_ml",   '"contact mechanics" "machine learning"'),
 ("mech_turbine_ml",   '"gas turbine" "machine learning" surrogate'),
 ("mech_composite_ml", '"composite" "machine learning" surrogate damage'),
 # ---------- 跨/通用 ----------
 ("gen_simlaw_break",  '"similarity" "breakdown" heat OR mechanics'),
 ("gen_scale_model",   '"scaling law" "machine learning" engineering'),
 ("gen_lowfi_hifi",    '"low-fidelity" "high-fidelity" correction neural'),
]
res = {}
for name, term in Q:
    d = s2(term)
    if "error" in d:
        print("[ERR ] %-20s %s" % (name, term)); res[name] = {"err": d["error"]}; time.sleep(4); continue
    tot = d.get("total", 0); data = d.get("data") or []
    res[name] = {"q": term, "total": tot, "top": [{"y":p.get("year"),"t":p.get("title"),"v":p.get("venue")} for p in data[:8]]}
    print("[%5s] %-20s %s" % (tot, name, term))
    for p in data[:4]:
        print("        - (%s) %s | %s" % (p.get("year"), p.get("title"), p.get("venue")))
    time.sleep(4)
json.dump(res, open(os.path.join(OUT, "s2_mech_veh.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved raw/s2_mech_veh.json")
