"""Family-cluster uncertainty for the 600-point random-wall calibration core.

The 27 low-contrast anchor points are not members of the 134 wall-mode
families in the 600-point random-wall sweep, so they are not resampled here.
They remain in the 627-point central fit in exp_eps_law.py.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_public" / "pairs600_fixedp.npz"
OUT = ROOT / "论文" / "cluster_boot.json"

z = np.load(DATA, allow_pickle=True)
X = z['X'].astype(np.float64); Y = z['Y'].astype(np.float64); meta = json.loads(str(z['meta']))
N = len(meta); eps = np.array([m['eps'] for m in meta]); D = Y - X
CH = ['ux', 'uy', 'theta', 'p']
R = np.stack([np.linalg.norm(D[:, c].reshape(N, -1), axis=1) /
              np.linalg.norm(Y[:, c].reshape(N, -1), axis=1) for c in range(4)], 1)
fam = [tuple(int(v) for v in m['nT']) for m in meta]
uf = {f: i for i, f in enumerate(sorted(set(fam)))}
cid = np.array([uf[f] for f in fam])
rng = np.random.default_rng(20261030); B = 5000
out = {
    "method": {
        "sample_count": int(N),
        "wall_families": int(len(uf)),
        "bootstrap_replicates": int(B),
        "resampling_unit": "wall-mode family (nT)",
        "interval": "95% percentile",
        "excluded_from_cluster_bootstrap": "27 low-contrast anchor points",
    },
    "channels": {},
}

for k, c in enumerate(CH):
    y = R[:, k]
    m0 = stats.linregress(np.log(eps), np.log(y)).slope
    q_design = np.column_stack([eps, eps**2, np.ones_like(eps)])
    q0, *_ = np.linalg.lstsq(q_design, y, rcond=None)
    kappa0 = q0[1] / q0[0]
    cl, iid, cluster_kappa = [], [], []
    for _ in range(B):
        pick = rng.choice(len(uf), len(uf), replace=True)
        idx = np.concatenate([np.where(cid == p)[0] for p in pick])
        cl.append(stats.linregress(np.log(eps[idx]), np.log(y[idx])).slope)
        j = rng.integers(0, N, N)
        iid.append(stats.linregress(np.log(eps[j]), np.log(y[j])).slope)
        q = np.linalg.lstsq(q_design[idx], y[idx], rcond=None)[0]
        cluster_kappa.append(q[1] / q[0])
    out["channels"][c] = {
        "power": {
            "m": float(m0),
            "cluster_ci95": list(map(float, np.percentile(cl, [2.5, 97.5]))),
            "iid_ci95": list(map(float, np.percentile(iid, [2.5, 97.5]))),
        },
        "quadratic": {
            "kappa": float(kappa0),
            "cluster_ci95": list(map(float, np.percentile(cluster_kappa, [2.5, 97.5]))),
        },
    }

OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
