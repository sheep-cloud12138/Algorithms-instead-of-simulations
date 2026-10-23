"""Average the per-sample field/Nu arrays across seeds and write a plot JSON."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "论文", "seed_sweep.json")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "后续方案", "field_vs_nu_3seed.json")
j = json.load(open(src))
seeds = sorted(int(k) for k in j["seeds"])
pre = json.load(open(os.path.join(ROOT, "后续方案", "prelim_field_vs_nu.json")))

def avg(kind, key):
    a = np.array([j["seeds"][str(s)][kind]["legacy"]["_" + key] for s in seeds])
    return a.mean(0)

t = {k: avg(k, "T") for k in ("pub_plain", "pub_prior", "pub_flux")}
n = {k: avg(k, "Nu") for k in ("pub_plain", "pub_prior", "pub_flux")}
rows = []
for i in range(len(t["pub_prior"])):
    rows.append({"i": i,
                 "floor_T": float(pre["rows"][i]["floor_T"]),
                 "floor_Nu": float(pre["rows"][i]["floor_Nu"]),
                 "prior_T": float(t["pub_prior"][i]), "prior_Nu": float(n["pub_prior"][i]),
                 "prior+flux_T": float(t["pub_flux"][i]),
                 "prior+flux_Nu": float(n["pub_flux"][i])})
json.dump({"n": len(rows), "seeds": seeds, "rows": rows}, open(dst, "w"), indent=1)
for k in ("pub_plain", "pub_prior", "pub_flux"):
    print(k, "T med %.4f Nu med %.4f frac %.3f" % (
        np.median(t[k]), np.median(n[k]), ((t[k] < .01) & (n[k] > .02)).mean()))
print("saved", dst)
