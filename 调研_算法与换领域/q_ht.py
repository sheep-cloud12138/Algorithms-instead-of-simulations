# -*- coding: utf-8 -*-
import time, json, os
from oa import q, slim
Q = [
 ("sco2_htd_ml",   '"supercritical" AND "heat transfer deterioration" AND ("machine learning" OR "neural network")', None),
 ("sco2_heat_ml",  '"supercritical CO2" AND ("machine learning" OR "neural network") AND "heat transfer"', None),
 ("htd",           '"heat transfer deterioration" AND "supercritical"', None),
 ("chf_ml",        '"critical heat flux" AND ("machine learning" OR "neural network")', None),
 ("boiling_ml",    '"flow boiling" AND ("machine learning" OR "neural network")', None),
 ("dittus",        '"Dittus-Boelter"', None),
 ("corr_ml_ht",    '"heat transfer correlation" AND ("machine learning" OR "neural network")', None),
 ("condensation_ml",'"condensation" AND ("machine learning" OR "neural network") AND "heat transfer"', None),
 ("microchannel_boil",'"microchannel" AND "flow boiling" AND ("machine learning" OR "neural network")', None),
 ("jet_imping_ml", '"impinging jet" AND "heat transfer" AND ("machine learning" OR "neural network")', None),
 ("fouling_ml",    '"fouling" AND "heat exchanger" AND ("machine learning" OR "neural network")', None),
 ("nusselt_ml",    '"Nusselt number" AND ("machine learning" OR "neural network") AND "correlation"', None),
 ("simlaw_ml",     '"similarity solution" AND ("machine learning" OR "neural network")', None),
 ("exponent_rec",  '"scaling exponent" AND ("machine learning" OR "neural network")', None),
]
res = {}
for name, term, y in Q:
    for a in range(6):
        d = q(term, per_page=5, year_from=y)
        if "error" not in d: break
        time.sleep(15)
    if "error" in d:
        print("[ERR ] %-18s %s" % (name, term)); res[name]={"err":1}; continue
    c = d["meta"]["count"]; res[name]={"term":term,"count":c}
    print("[%6d] %-18s %s" % (c, name, term))
    for w in d["results"][:4]:
        h = slim(w); print("          - (%s) %s [%s] c=%s" % (h["y"], h["t"], h["v"], h["c"]))
    time.sleep(4)
json.dump(res, open("raw/ht_gap.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
