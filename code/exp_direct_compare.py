"""Richardson combination of the direct linear response vs the measured D/eps."""
import os, sys, json
import numpy as np

BASE = os.path.expanduser("~/nc_run/paired/dataset")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct")
EPS = [0.4, 0.8, 1.2]


def load(d, n):
    return np.load(os.path.join(d, n + ".npy")).astype(np.float64)


def deramp(v, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ c


def case_dir(case, eps):
    return os.path.join(BASE, "%s_e%.1f" % (case, eps), "sample_000000")


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "c00_Ra1e2_1100"
    Pd = {}; Rd = {}
    for e in EPS:
        d = case_dir(case, e)
        z = np.load(os.path.join(OUT, "%s_e%.1f.npz" % (case, e)))
        y = load(d, "coords_p_bouss")[:, 1]
        du = z["du"].reshape(-1, 2) / e
        de = z["de"] / e
        dp = deramp(z["dp"], y) / e
        ux = load(d, "ux_comp") - load(d, "ux_bouss")
        uy = load(d, "uy_comp") - load(d, "uy_bouss")
        dth = load(d, "T_comp") - load(d, "T_bouss")
        dpc = deramp(load(d, "p_comp_pert_raw") + load(d, "p_bouss_pert_raw"), y)
        Pd[e] = {"ux": du[:, 0], "uy": du[:, 1], "theta": de, "p": dp}
        Rd[e] = {"ux": ux / e, "uy": uy / e, "theta": dth / e, "p": dpc / e}

    for ch in ("ux", "uy", "theta", "p"):
        pd = 3 * Pd[0.4][ch] - 3 * Pd[0.8][ch] + Pd[1.2][ch]
        rd = 3 * Rd[0.4][ch] - 3 * Rd[0.8][ch] + Rd[1.2][ch]
        err = np.linalg.norm(pd - rd) / np.linalg.norm(rd)
        print("%-5s |P_direct|=%.4e |P_data|=%.4e relerr=%.3f corr=%.4f" %
              (ch, np.linalg.norm(pd), np.linalg.norm(rd), err, np.corrcoef(pd, rd)[0, 1]))
    print()
    for e in EPS:
        row = []
        for ch in ("ux", "uy", "theta", "p"):
            err = np.linalg.norm(Pd[e][ch] - Rd[e][ch]) / np.linalg.norm(Rd[e][ch])
            row.append("%s=%.3f" % (ch, err))
        print("eps=%.1f  " % e + "  ".join(row))


if __name__ == "__main__":
    main()
