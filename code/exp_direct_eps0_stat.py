import json, os
import numpy as np
d = json.load(open(os.path.expanduser("~/nc_run/../../mnt/d/Desktop/算法代替仿真/论文/direct_eps0/summary.json"))) if False else None
R = json.load(open("/mnt/d/Desktop/算法代替仿真/论文/direct_eps0/summary.json"))
for c in ("ux", "uy", "theta", "p"):
    e = np.array([r[c] for r in R]); k = np.array([r[c + "_cos"] for r in R])
    print("%-5s relerr med %.3f p10 %.3f p90 %.3f  cos med %.4f min %.4f" %
          (c, np.median(e), *np.percentile(e, [10, 90]), np.median(k), k.min()))
bys = {}
for r in R:
    bys.setdefault(r["eps"], []).append(r)
for eps in sorted(bys):
    print("eps %.2f n=%d " % (eps, len(bys[eps])) + "  ".join(
        "%s %.3f" % (c, np.median([x[c] for x in bys[eps]])) for c in ("ux", "uy", "theta", "p")))