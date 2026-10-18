"""Aggregate the direct linear response against the data over all configurations."""
import os, sys, glob, json
import numpy as np

BASE = os.path.expanduser("~/nc_run/paired/dataset")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct")
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct_summary.json")
EPS3 = [0.4, 0.8, 1.2]
CH = ["ux", "uy", "theta", "p"]


def rd(d, n):
    return np.load(os.path.join(d, n + ".npy")).astype(np.float64)


def deramp(v, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ c


def channels(case, eps):
    d = os.path.join(BASE, "%s_e%.1f" % (case, eps), "sample_000000")
    z = np.load(os.path.join(OUT, "%s_e%.1f.npz" % (case, eps)))
    y = rd(d, "coords_p_bouss")[:, 1]
    du = z["du"].reshape(-1, 2) / eps
    P = {"ux": du[:, 0], "uy": du[:, 1], "theta": z["de"] / eps,
         "p": deramp(z["dp"], y) / eps}
    dth = rd(d, "T_comp") - rd(d, "T_bouss")
    D = {"ux": (rd(d, "ux_comp") - rd(d, "ux_bouss")) / eps,
         "uy": (rd(d, "uy_comp") - rd(d, "uy_bouss")) / eps,
         "theta": dth / eps,
         "p": deramp(rd(d, "p_comp_pert_raw") + rd(d, "p_bouss_pert_raw"), y) / eps}
    return P, D


def main():
    cases = sorted({os.path.basename(p).rsplit("_e", 1)[0]
                    for p in glob.glob(os.path.join(BASE, "c*_e0.4"))})
    rows = []
    for case in cases:
        try:
            P, D = {}, {}
            for e in EPS3:
                p, d = channels(case, e)
                for c in CH:
                    P.setdefault(c, {})[e] = p[c]
                    D.setdefault(c, {})[e] = d[c]
            rec = {"case": case}
            for c in CH:
                pr = 3 * P[c][0.4] - 3 * P[c][0.8] + P[c][1.2]
                dr = 3 * D[c][0.4] - 3 * D[c][0.8] + D[c][1.2]
                rec[c] = dict(relerr=float(np.linalg.norm(pr - dr) / np.linalg.norm(dr)),
                              corr=float(np.corrcoef(pr, dr)[0, 1]),
                              npred=float(np.linalg.norm(pr)), ndata=float(np.linalg.norm(dr)))
            rows.append(rec)
        except Exception as ex:
            print("skip", case, ex)
    print("n =", len(rows))
    for c in CH:
        e = np.array([r[c]["relerr"] for r in rows]); k = np.array([r[c]["corr"] for r in rows])
        print("%-5s relerr median %.3f (p10-p90 %.3f-%.3f)  corr median %.4f (min %.4f)" %
              (c, np.median(e), np.percentile(e, 10), np.percentile(e, 90), np.median(k), k.min()))
    json.dump(rows, open(RES, "w"), indent=1)
    print("saved", RES)


if __name__ == "__main__":
    main()
