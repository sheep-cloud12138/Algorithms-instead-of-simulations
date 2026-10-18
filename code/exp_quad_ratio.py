"""Measured quadratic-to-linear ratio q = |Q|/|P| per configuration."""
import os, glob, json
import numpy as np
import exp_direct_pereps as M

def rich(f, w):
    a, b, c = w
    return 3 * f[a] - 3 * f[b] + f[c]

cases = sorted({os.path.basename(p).rsplit("_e", 1)[0]
                for p in glob.glob(os.path.join(M.BASE, "c*_e0.4"))})
out = {}
for c in M.CH:
    q = []
    for k in cases:
        D = {e: M.pair(k, e)[1][c] for e in (0.4, 0.8, 1.2, 1.9)}
        P = rich(D, (0.4, 0.8, 1.2))
        e = 1.9
        q.append(float(np.linalg.norm(D[e] - P) / (e * np.linalg.norm(P))))
    out[c] = [float(np.median(q)), float(np.percentile(q, 10)), float(np.percentile(q, 90))]
    print("%-5s |Q|/|P| median %.2f (p10 %.2f p90 %.2f)" % (c, *out[c]))
json.dump(out, open(os.path.join(os.path.dirname(M.OUT), "quad_ratio.json"), "w"), indent=1)