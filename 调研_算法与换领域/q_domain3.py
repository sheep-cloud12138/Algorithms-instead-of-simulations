import time
from oa import q, slim
Q = [
 ("subsurface_ml", '"subsurface flow" AND ("machine learning" OR "surrogate" OR "neural operator")'),
 ("co2_ml",        '"CO2 storage" AND ("machine learning" OR "surrogate" OR "neural operator")'),
 ("photonic_ml",   '"photonics" AND ("neural network" OR "surrogate" OR "deep learning")'),
 ("em_surrogate",  '"electromagnetic" AND "surrogate model"'),
 ("nonnewton_ml",  '"non-Newtonian" AND ("machine learning" OR "surrogate")'),
 ("corr_empirical",'"empirical correlation" AND ("machine learning" OR "neural network")'),
 ("model_discrep", '"model discrepancy" AND ("machine learning" OR "Gaussian process")'),
]
for name, term in Q:
    for attempt in range(5):
        d = q(term, per_page=3)
        if "error" not in d: break
        time.sleep(20)
    if "error" in d:
        print("[ERR ] %-16s %s" % (name, term)); continue
    print("[%6d] %-16s %s" % (d["meta"]["count"], name, term))
    for w in d["results"]:
        h = slim(w); print("          - (%s) %s [%s] c=%s" % (h["y"], h["t"], h["v"], h["c"]))
    time.sleep(5)
