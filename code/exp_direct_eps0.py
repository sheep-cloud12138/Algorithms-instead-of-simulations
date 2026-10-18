"""Direct linear response on the eps=0.02/0.05/0.10 sweep (genuinely small contrast)."""
import os, glob, json, sys
import numpy as np
import ufl
from dolfinx import fem
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_direct_linear as M

ROOT = os.path.expanduser("~/nc_run/eps0/dataset")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct_eps0")
CH = ["ux", "uy", "theta", "p"]


def rd(d, n):
    return np.load(os.path.join(d, n + ".npy")).astype(np.float64)


def deramp(v, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ c


def solve(d, meta):
    eps = float(meta["dT"]); Ra = float(meta["Ra"]); res = int(meta["res"])
    L = float(meta["length"]); Ma = float(meta["Ma"])
    msh = M.G.create_mesh(L, res); W = M.G.space_comp(msh)
    bcs = M.make_bcs(msh, W, meta, L)
    wx = M.build_X(msh, W, d, eps)
    F = M.residual(msh, W, wx, Ra, eps, Ma)
    Jf = ufl.replace(ufl.derivative(F, wx), {wx: wx})
    prob = fem.petsc.LinearProblem(
        Jf, -ufl.replace(F, {wx: wx}), bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    delta = prob.solve()
    return (delta.sub(0).collapse().x.array.copy(), delta.sub(1).collapse().x.array.copy(),
            delta.sub(2).collapse().x.array.copy(), delta.sub(3).collapse().x.array.copy())


def main():
    dirs = sorted(glob.glob(os.path.join(ROOT, "sample_*")))
    rows = []
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        eps = float(meta["dT"])
        try:
            du, de, dr, dp = solve(d, meta)
        except Exception as ex:
            print("FAIL", d, type(ex).__name__, ex); continue
        y = rd(d, "coords_p_bouss")[:, 1]
        dur = du.reshape(-1, 2)
        P = {"ux": dur[:, 0] / eps, "uy": dur[:, 1] / eps, "theta": de / eps,
             "p": deramp(dp, y) / eps}
        D = {"ux": (rd(d, "ux_comp") - rd(d, "ux_bouss")) / eps,
             "uy": (rd(d, "uy_comp") - rd(d, "uy_bouss")) / eps,
             "theta": (rd(d, "T_comp") - rd(d, "T_bouss")) / eps,
             "p": deramp(rd(d, "p_comp_pert_raw") + rd(d, "p_bouss_pert_raw"), y) / eps}
        rec = {"dir": os.path.basename(d), "Ra": meta["Ra"], "eps": eps}
        for c in CH:
            rec[c] = float(np.linalg.norm(P[c] - D[c]) / np.linalg.norm(D[c]))
            rec[c + "_r"] = float(np.linalg.norm(P[c]) / np.linalg.norm(D[c]))
            rec[c + "_cos"] = float(np.dot(P[c], D[c]) /
                                    (np.linalg.norm(P[c]) * np.linalg.norm(D[c])))
        rows.append(rec)
        print("Ra %-7.0f eps %.2f  " % (meta["Ra"], eps) +
              "  ".join("%s %.3f (norm %.2f, cos %.3f)" % (c, rec[c], rec[c + "_r"], rec[c + "_cos"])
                        for c in CH), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print()
    for c in CH:
        v = np.array([r[c] for r in rows])
        print("%-5s relerr median %.3f  p90 %.3f" % (c, np.median(v), np.percentile(v, 90)))


if __name__ == "__main__":
    main()