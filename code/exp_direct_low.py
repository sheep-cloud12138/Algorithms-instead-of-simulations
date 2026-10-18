"""Direct linear response at eps=0.1/0.2 on the paired configurations (own state, no rebuild)."""
import os, glob, json
import numpy as np
import ufl
from dolfinx import fem
import exp_direct_linear as M

ROOT = os.path.expanduser("~/nc_run/paired/loweps")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct_low")
EPS = [0.1, 0.2]


def solve(d, meta, eps):
    Ra = float(meta["Ra"]); res = int(meta["res"]); L = float(meta["length"]); Ma = float(meta["Ma"])
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
    return dict(du=delta.sub(0).collapse().x.array.copy(),
                de=delta.sub(1).collapse().x.array.copy(),
                dr=delta.sub(2).collapse().x.array.copy(),
                dp=delta.sub(3).collapse().x.array.copy())


def main():
    os.makedirs(OUT, exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(ROOT, "c*_e0.1")))
    for d in dirs:
        case = os.path.basename(d).rsplit("_e", 1)[0]
        for eps in EPS:
            dd = os.path.join(ROOT, "%s_e%.1f" % (case, eps), "sample_000000")
            f = os.path.join(OUT, "%s_e%.1f.npz" % (case, eps))
            if os.path.exists(f) or not os.path.isdir(dd):
                continue
            try:
                meta = json.load(open(os.path.join(dd, "meta.json")))
                np.savez(f, **solve(dd, meta, eps))
                print("ok", case, eps, flush=True)
            except Exception as ex:
                print("FAIL", case, eps, type(ex).__name__, ex, flush=True)


if __name__ == "__main__":
    main()