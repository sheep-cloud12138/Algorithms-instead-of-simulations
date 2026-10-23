"""Common-exponent fit for the Ra=1e5 3-D set (three flow intercepts)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAW = os.path.join(ROOT, "论文", "law3d_ra1e5.json")
OUT = os.path.join(ROOT, "论文", "3d_ra1e5_summary.json")


def main():
    rows = json.load(open(LAW))
    flows = {}
    for r in rows:
        flows.setdefault(int(r["flow"]), []).append(r)
    flows = {f: r for f, r in flows.items() if len({x["eps"] for x in r}) >= 3}
    fs = sorted(flows)
    out = {"n_total": len(rows), "n_paired": sum(len(r) for r in flows.values()),
           "n_flows": len(fs), "Ra": 1e5, "channels": {}}
    for ch in ("ux", "uy", "theta", "p"):
        y, X = [], []
        for f in fs:
            sub = sorted(flows[f], key=lambda z: z["eps"])
            for r in sub:
                y.append(np.log(r[ch]))
                row = [np.log(r["eps"])] + [1.0 if f == g else 0.0 for g in fs]
                X.append(row)
        y, X = np.array(y), np.array(X)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        dof = max(1, len(y) - X.shape[1])
        s2 = float(res @ res) / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        ss = float(((y - y.mean()) ** 2).sum())
        out["channels"][ch] = {
            "m": float(coef[0]), "se": float(np.sqrt(cov[0, 0])),
            "A_per_flow": [float(np.exp(c)) for c in coef[1:]],
            "R2": float(1 - (res @ res) / ss),
        }
        print("%-5s m=%.3f+-%.3f A=%s R2=%.3f" % (
            ch, coef[0], np.sqrt(cov[0, 0]),
            "/".join("%.4f" % np.exp(c) for c in coef[1:]),
            1 - (res @ res) / ss))
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
