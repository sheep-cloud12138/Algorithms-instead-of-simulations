"""Batch: direct linear response on all paired configurations."""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_direct_linear as M
import ufl
from dolfinx import fem

BASE = os.path.expanduser("~/nc_run/paired/dataset")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct")
EPS3 = [0.4, 0.8, 1.2, 1.6, 1.9]
EPS5 = [0.4, 0.8, 1.2, 1.6, 1.9]


def rd(d, n):
    return np.load(os.path.join(d, n + ".npy")).astype(np.float64)


def solve_one(case, eps):
    d = os.path.join(BASE, "%s_e%.1f" % (case, eps), "sample_000000")
    meta = json.load(open(os.path.join(d, "meta.json")))
    Ra = float(meta["Ra"]); res = int(meta["res"]); L = float(meta["length"]); Ma = float(meta["Ma"])
    msh = M.G.create_mesh(L, res); W = M.G.space_comp(msh)
    bcs = M.make_bcs(msh, W, meta, L)
    wx = M.build_X(msh, W, d, eps)
    F = M.residual(msh, W, wx, Ra, eps, Ma)
    Lf = -ufl.replace(F, {wx: wx})
    Jf = ufl.replace(ufl.derivative(F, wx), {wx: wx})
    prob = fem.petsc.LinearProblem(
        Jf, Lf, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    delta = prob.solve()
    return dict(du=delta.sub(0).collapse().x.array.copy(),
                de=delta.sub(1).collapse().x.array.copy(),
                dr=delta.sub(2).collapse().x.array.copy(),
                dp=delta.sub(3).collapse().x.array.copy())


def main():
    cases = sorted({os.path.basename(p).rsplit("_e", 1)[0]
                    for p in glob.glob(os.path.join(BASE, "c*_e0.4"))})
    print("cases", len(cases))
    for case in cases:
        for eps in EPS3:
            f = os.path.join(OUT, "%s_e%.1f.npz" % (case, eps))
            if os.path.exists(f):
                continue
            try:
                z = solve_one(case, eps)
                np.savez(f, **z)
                print("ok", case, eps, flush=True)
            except Exception as ex:
                print("FAIL", case, eps, type(ex).__name__, ex, flush=True)


if __name__ == "__main__":
    main()

