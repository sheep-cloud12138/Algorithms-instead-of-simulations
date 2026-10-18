"""Held-out check: P from eps=0.4/0.8/1.2 predicts measured D/eps at eps=0.1 and 0.2."""
import os, glob, json
import numpy as np

BASE = os.path.expanduser("~/nc_run/paired/dataset")
LOW = os.path.expanduser("~/nc_run/paired/loweps")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct")
CH = ["ux", "uy", "theta", "p"]


def rd(d, n):
    return np.load(os.path.join(d, n + ".npy")).astype(np.float64)


def deramp(v, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ c


def data(root, case, eps):
    d = os.path.join(root, "%s_e%.1f" % (case, eps), "sample_000000")
    y = rd(d, "coords_p_bouss")[:, 1]
    return {"ux": (rd(d, "ux_comp") - rd(d, "ux_bouss")) / eps,
            "uy": (rd(d, "uy_comp") - rd(d, "uy_bouss")) / eps,
            "theta": (rd(d, "T_comp") - rd(d, "T_bouss")) / eps,
            "p": deramp(rd(d, "p_comp_pert_raw") + rd(d, "p_bouss_pert_raw"), y) / eps}


def npzf(case, eps):
    for tag in ("%.2f", "%.1f"):
        f = os.path.join(OUT, (case + "_e" + tag + ".npz") % eps)
        if os.path.exists(f):
            return np.load(f)
    raise FileNotFoundError(case)


def direct(case, eps):
    d = os.path.join(BASE, "%s_e0.4" % case, "sample_000000")
    z = npzf(case, eps)
    y = rd(d, "coords_p_bouss")[:, 1]
    du = z["du"].reshape(-1, 2) / eps
    return {"ux": du[:, 0], "uy": du[:, 1], "theta": z["de"] / eps, "p": deramp(z["dp"], y) / eps}


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    cases = sorted({os.path.basename(p).rsplit("_e", 1)[0]
                    for p in glob.glob(os.path.join(BASE, "c*_e0.4"))})
    rows = []
    for case in cases:
        try:
            P = {e: data(BASE, case, e) for e in (0.4, 0.8, 1.2)}
            L = {e: data(LOW, case, e) for e in (0.1, 0.2)}
        except FileNotFoundError:
            print("skip", case)
            continue
        D1 = {e: direct(case, e) for e in (0.1, 0.2)}
        rec = {"case": case}
        for c in CH:
            pw = 3 * P[0.4][c] - 3 * P[0.8][c] + P[1.2][c]
            rec[c] = dict(
                p_at_01=rel(pw, L[0.1][c]),
                p_at_02=rel(pw, L[0.2][c]),
                direct_at_01=rel(D1[0.1][c], L[0.1][c]),
                direct_at_02=rel(D1[0.2][c], L[0.2][c]),
                ramp=rel(L[0.1][c], L[0.2][c]),
                p_vs_04=rel(pw, P[0.4][c]),
            )
        rows.append(rec)
    hdr = ["P->D(0.1)", "P->D(0.2)", "lin->D(0.1)", "lin->D(0.2)", "D(0.1) vs D(0.2)", "P vs D(0.4)"]
    print("%-6s" % "" + "".join("%16s" % h for h in hdr))
    for c in CH:
        ks = ("p_at_01", "p_at_02", "direct_at_01", "direct_at_02", "ramp", "p_vs_04")
        print("%-6s" % c + "".join("%16.3f" % np.median([r[c][k] for r in rows]) for k in ks))
    json.dump(rows, open(os.path.join(os.path.dirname(OUT), "heldout_loweps.json"), "w"), indent=1)


if __name__ == "__main__":
    main()