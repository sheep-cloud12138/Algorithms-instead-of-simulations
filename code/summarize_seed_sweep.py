"""Print paper-ready summaries from 论文/seed_sweep.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                   os.path.join(os.path.dirname(HERE), "论文", "seed_sweep.json")))
seeds = sorted(int(k) for k in J["seeds"])
print("seeds:", seeds)

for split in ("extrap", "interp"):
    print("\n==", split)
    for tag, kind in (("plain", "pub_plain"), ("plain_rescaled", "pub_plain"),
                      ("prior", "pub_prior")):
        v = np.array([J["seeds"][str(s)][kind][split][tag] for s in seeds])
        m, sd = v.mean(0), v.std(0, ddof=1) if len(v) > 1 else np.zeros(4)
        print("%-16s %s  avg %.4f" % (
            tag, " ".join("%.4f±%.4f" % (a, b) for a, b in zip(m, sd)), m.mean()))

print("\n== legacy per-sample medians/means")
for kind in ("pub_plain", "pub_prior", "pub_flux"):
    ts = [J["seeds"][str(s)][kind]["legacy"]["_T"] for s in seeds]
    ns = [J["seeds"][str(s)][kind]["legacy"]["_Nu"] for s in seeds]
    ls = [J["seeds"][str(s)][kind]["legacy"]["_local"] for s in seeds]
    T, N, L = map(lambda x: np.array(x), (ts, ns, ls))
    fr = [((t < .01) & (n > .02)).mean() for t, n in zip(T, N)]
    print("%-10s T med %.4f (seed-mean; %.4f±%.4f) | Nu med %.4f (%.4f±%.4f) | local mean %.4f±%.4f | frac %.3f±%.3f"
          % (kind, np.median(T.mean(0)), T.mean(1).mean(), T.mean(1).std(ddof=1),
             np.median(N.mean(0)), N.mean(1).mean(), N.mean(1).std(ddof=1),
             L.mean(1).mean(), L.mean(1).std(ddof=1),
             np.mean(fr), np.std(fr, ddof=1)))

if len(seeds) > 1:
    a = np.array([J["seeds"][str(s)]["pub_prior"]["legacy"]["_Nu"] for s in seeds]).mean(0)
    b = np.array([J["seeds"][str(s)]["pub_flux"]["legacy"]["_Nu"] for s in seeds]).mean(0)
    d = a - b
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
    print("prior-flux Nu diff %.4f  95%% CI [%.4f, %.4f]" %
          (d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
