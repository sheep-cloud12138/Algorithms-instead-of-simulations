"""Per-eps agreement: direct linear response vs measured difference, same eps."""
import os, glob, json
import numpy as np

BASE = os.path.expanduser("~/nc_run/paired/dataset")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct")
CH = ["ux", "uy", "theta", "p"]
DATA = [0.4, 0.8, 1.2, 1.6, 1.9]


def rd(d, n):
    return np.load(os.path.join(d, n + ".npy")).astype(np.float64)


def deramp(v, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ c


def npz(case, eps):
    for tag in ("%.2f", "%.1f"):
        f = os.path.join(OUT, (case + "_e" + tag + ".npz") % eps)
        if os.path.exists(f):
            return np.load(f)
    raise FileNotFoundError(case)


def pair(case, eps):
    d = os.path.join(BASE, "%s_e%.1f" % (case, eps), "sample_000000")
    z = npz(case, eps)
    y = rd(d, "coords_p_bouss")[:, 1]
    du = z["du"].reshape(-1, 2) / eps
    P = {"ux": du[:, 0], "uy": du[:, 1], "theta": z["de"] / eps, "p": deramp(z["dp"], y) / eps}
    D = {"ux": (rd(d, "ux_comp") - rd(d, "ux_bouss")) / eps,
         "uy": (rd(d, "uy_comp") - rd(d, "uy_bouss")) / eps,
         "theta": (rd(d, "T_comp") - rd(d, "T_bouss")) / eps,
         "p": deramp(rd(d, "p_comp_pert_raw") + rd(d, "p_bouss_pert_raw"), y) / eps}
    return P, D


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    cases = sorted({os.path.basename(p).rsplit("_e", 1)[0]
                    for p in glob.glob(os.path.join(BASE, "c*_e0.4"))})
    out = {}
    for eps in DATA:
        rows = [pair(k, eps) for k in cases]
        out[eps] = {c: [rel(P[c], D[c]) for P, D in rows] for c in CH}
    print("%-6s" % "eps" + "".join("%24s" % c for c in CH))
    for eps in DATA:
        print("%-6.1f" % eps + "".join("%24s" % ("%.3f (%.3f-%.3f)" % tuple(np.percentile(out[eps][c], [50, 10, 90])))
                                       for c in CH))
    print()
    print("range of D/eps norm versus eps (median over cases, normalised to eps=0.4)")
    for c in CH:
        n = [np.median([np.linalg.norm(pair(k, e)[1][c]) for k in cases]) for e in DATA]
        print("  %-5s " % c + " ".join("%.1f:%.3f" % (e, v / n[0]) for e, v in zip(DATA, n)))
    json.dump({str(k): {c: list(map(float, v)) for c, v in d.items()} for k, d in out.items()},
              open(os.path.join(os.path.dirname(OUT), "direct_pereps.json"), "w"), indent=1)


if __name__ == "__main__":
    main()