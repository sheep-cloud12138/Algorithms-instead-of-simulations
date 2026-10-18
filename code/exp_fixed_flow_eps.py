"""Fixed-flow eps-dependence of the one-Newton-step response and of the measured difference."""
import os, glob, json
import numpy as np

BASE = os.path.expanduser("~/nc_run/paired/dataset")
LOW = os.path.expanduser("~/nc_run/paired/loweps")
DL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct")
DLO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct_low")
CH = ["ux", "uy", "theta", "p"]


def rd(d, n):
    return np.load(os.path.join(d, n + ".npy")).astype(np.float64)


def deramp(v, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ c


def load_pair(root, case, eps):
    d = os.path.join(root, "%s_e%.1f" % (case, eps), "sample_000000")
    y = rd(d, "coords_p_bouss")[:, 1]
    return {"ux": (rd(d, "ux_comp") - rd(d, "ux_bouss")) / eps,
            "uy": (rd(d, "uy_comp") - rd(d, "uy_bouss")) / eps,
            "theta": (rd(d, "T_comp") - rd(d, "T_bouss")) / eps,
            "p": deramp(rd(d, "p_comp_pert_raw") + rd(d, "p_bouss_pert_raw"), y) / eps}


def load_direct(root, case, eps):
    z = np.load(os.path.join(root, "%s_e%.1f.npz" % (case, eps)))
    d = os.path.join(LOW, "%s_e0.1" % case, "sample_000000")
    y = rd(d, "coords_p_bouss")[:, 1]
    du = z["du"].reshape(-1, 2) / eps
    return {"ux": du[:, 0], "uy": du[:, 1], "theta": z["de"] / eps, "p": deramp(z["dp"], y) / eps}


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    cases = sorted({os.path.basename(p).rsplit("_e", 1)[0]
                    for p in glob.glob(os.path.join(LOW, "c*_e0.1"))})
    rows = []
    for case in cases:
        try:
            dl = {e: load_direct(DLO, case, e) for e in (0.1, 0.2)}
            dl.update({e: load_direct(DL, case, e) for e in (0.4, 1.9)})
            dt = {e: load_pair(LOW, case, e) for e in (0.1, 0.2)}
            dt.update({e: load_pair(BASE, case, e) for e in (0.4, 1.9)})
        except FileNotFoundError:
            continue
        rec = {"case": case}
        for c in CH:
            rec[c] = dict(d01_vs_04=rel(dl[0.1][c], dl[0.4][c]),
                          d02_vs_04=rel(dl[0.2][c], dl[0.4][c]),
                          d19_vs_04=rel(dl[1.9][c], dl[0.4][c]),
                          D01_vs_04=rel(dt[0.1][c], dt[0.4][c]),
                          D19_vs_04=rel(dt[1.9][c], dt[0.4][c]),
                          lin01_vs_D01=rel(dl[0.1][c], dt[0.1][c]))
        rows.append(rec)
    print("n =", len(rows))
    ks = ("d01_vs_04", "d02_vs_04", "d19_vs_04", "D01_vs_04", "D19_vs_04", "lin01_vs_D01")
    hdr = ["d(0.1)/d(0.4)", "d(0.2)/d(0.4)", "d(1.9)/d(0.4)", "D(0.1)/D(0.4)", "D(1.9)/D(0.4)", "d vs D at 0.1"]
    print("%-6s" % "" + "".join("%17s" % h for h in hdr))
    for c in CH:
        print("%-6s" % c + "".join("%17.3f" % np.median([r[c][k] for r in rows]) for k in ks))
    json.dump(rows, open(os.path.join(os.path.dirname(DL), "fixed_flow_eps.json"), "w"), indent=1)


if __name__ == "__main__":
    main()