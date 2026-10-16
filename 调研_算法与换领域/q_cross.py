import time, json
from oa import q, slim
Q=[
 ("no_solid",   '"neural operator" AND ("solid mechanics" OR "elasticity" OR "hyperelasticity")'),
 ("fno_mech",   '"Fourier neural operator" AND ("mechanics" OR "structural analysis" OR "elasticity")'),
 ("no_contact", '"neural operator" AND ("contact mechanics" OR "tribology" OR "friction")'),
 ("no_fatigue", '"neural operator" AND "fatigue"'),
 ("no_frac",    '"neural operator" AND "fracture"'),
 ("archard_ml", '"Archard" AND ("machine learning" OR "neural network")'),
 ("tool_err_ml",'"thermal error" AND "machine tool" AND ("machine learning" OR "neural network")'),
 ("batt_soh",   '"state of health" AND "lithium-ion battery" AND ("machine learning" OR "deep learning")'),
 ("crash_ml",   '"crashworthiness" AND ("machine learning" OR "surrogate")'),
 ("veh_nvh",    '"NVH" AND ("machine learning" OR "deep learning")'),
 ("larson_ml",  '"Larson-Miller" AND ("machine learning" OR "neural network")'),
 ("fno_ht2",    '"Fourier neural operator" AND "heat transfer"'),
]
for name, term in Q:
    for a in range(5):
        d=q(term, per_page=4)
        if "error" not in d: break
        time.sleep(12)
    if "error" in d:
        print("[ERR ] %-14s %s"%(name,term)); continue
    print("[%6d] %-14s %s"%(d["meta"]["count"],name,term))
    for w in d["results"][:3]:
        h=slim(w); print("         - (%s) %s [%s]"%(h["y"],h["t"],h["v"]))
    time.sleep(3)
