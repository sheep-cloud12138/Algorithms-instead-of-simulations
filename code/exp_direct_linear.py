"""Direct O(eps) response: solve J delta = -F(X_ext) and compare with Y - X."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/nc_run/DataGeneration"))
import genData_stochastic_fourierBC as G
from petsc4py import PETSc
import ufl
from dolfinx import fem

PR = 0.71
GAMMA = 1.4


def load(d, n):
    return np.load(os.path.join(d, n + ".npy")).astype(np.float64)


def deramp(v, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ c


def residual(msh, W, w, Ra, eps, Ma):
    u, e, rho, p = ufl.split(w)
    v, we, wr, q = ufl.TestFunctions(W)
    g = fem.Constant(msh, np.array([0.0, -1.0]))
    dxm = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 4})
    Mc = fem.Constant(msh, PETSc.ScalarType(Ma))
    Re = fem.Constant(msh, PETSc.ScalarType(np.sqrt(Ra / PR)))
    Fr = fem.Constant(msh, PETSc.ScalarType(np.sqrt(eps)))
    sigma = (ufl.nabla_grad(u) + ufl.nabla_grad(u).T
             - (2.0 / 3.0) * ufl.div(u) * ufl.Identity(msh.geometry.dim))
    E = e + 0.5 * GAMMA * (GAMMA - 1.0) * Mc * Mc * ufl.dot(u, u)
    P_work = (GAMMA - 1.0) * (GAMMA * Mc * Mc * p + 1.0)
    return (
        ufl.dot(ufl.div(rho * u), q) * dxm
        + ufl.dot(ufl.div(ufl.outer(rho * u, u)), v) * dxm
        - p * ufl.div(v) * dxm
        + (1.0 / Re) * ufl.inner(sigma, ufl.nabla_grad(v)) * dxm
        - (1.0 / (Fr * Fr)) * ufl.dot(rho * g, v) * dxm
        + ufl.dot(ufl.div(rho * E * u), we) * dxm
        + (GAMMA / (Re * PR)) * ufl.dot(ufl.grad(e), ufl.grad(we)) * dxm
        - ufl.dot(P_work * u, ufl.grad(we)) * dxm
        + (GAMMA * (GAMMA - 1.0) * Mc * Mc / Re) * ufl.inner(ufl.dot(sigma, u), ufl.grad(we)) * dxm
        - (GAMMA * (GAMMA - 1.0) * Mc * Mc / (Fr * Fr)) * ufl.dot(rho * g, u) * we * dxm
        + ((GAMMA * Mc * Mc * p + 1.0) - rho * e) * wr * dxm
    )


def make_bcs(msh, W, meta, L):
    Vu, _ = W.sub(0).collapse(); Ve, _ = W.sub(1).collapse()
    fdim, w_all, wL, wR, wB, wT = G.locate_walls(msh, L)
    u0 = fem.Function(Vu); u0.x.array[:] = 0.0
    bcs = [fem.dirichletbc(u0, fem.locate_dofs_topological((W.sub(0), Vu), fdim, w_all), W.sub(0))]
    e0 = fem.Function(Ve); e0.x.array[:] = 0.0
    walls = meta["bc_comp"]["walls"]
    for name, ents in (("left", wL), ("right", wR), ("bottom", wB), ("top", wT)):
        if walls[name]["type"].startswith("fourier") or walls[name]["type"] == "constant":
            bcs.append(fem.dirichletbc(e0, fem.locate_dofs_topological((W.sub(1), Ve), fdim, ents), W.sub(1)))
    return bcs


def build_X(msh, W, d, eps):
    Vu, _ = W.sub(0).collapse(); Ve, _ = W.sub(1).collapse()
    Vr, _ = W.sub(2).collapse(); Vp, _ = W.sub(3).collapse()
    w = fem.Function(W)
    uf = fem.Function(Vu)
    uf.x.array.reshape(-1, 2)[:, 0] = load(d, "ux_bouss")
    uf.x.array.reshape(-1, 2)[:, 1] = load(d, "uy_bouss")
    w.sub(0).interpolate(uf)
    T = load(d, "T_bouss")
    Tf = fem.Function(Ve); Tf.x.array[:] = T; w.sub(1).interpolate(Tf)
    rf = fem.Function(Vr); rf.x.array[:] = 1.0 / T; w.sub(2).interpolate(rf)
    p = -load(d, "p_bouss_raw") - load(d, "coords_p_bouss")[:, 1] / eps
    pf = fem.Function(Vp); pf.x.array[:] = p; w.sub(3).interpolate(pf)
    return w


def build_Y(msh, W, d):
    Vu, _ = W.sub(0).collapse(); Ve, _ = W.sub(1).collapse()
    Vr, _ = W.sub(2).collapse(); Vp, _ = W.sub(3).collapse()
    w = fem.Function(W)
    uf = fem.Function(Vu)
    uf.x.array.reshape(-1, 2)[:, 0] = load(d, "ux_comp")
    uf.x.array.reshape(-1, 2)[:, 1] = load(d, "uy_comp")
    w.sub(0).interpolate(uf)
    Tf = fem.Function(Ve); Tf.x.array[:] = load(d, "T_comp"); w.sub(1).interpolate(Tf)
    rf = fem.Function(Vr); rf.x.array[:] = load(d, "rho_comp"); w.sub(2).interpolate(rf)
    pf = fem.Function(Vp); pf.x.array[:] = load(d, "p_comp_raw"); w.sub(3).interpolate(pf)
    return w


def main():
    case = sys.argv[1]; eps = float(sys.argv[2])
    d = os.path.join(os.path.expanduser("~/nc_run/paired/dataset"), "%s_e%.1f" % (case, eps), "sample_000000")
    meta = json.load(open(os.path.join(d, "meta.json")))
    Ra = float(meta["Ra"]); res = int(meta["res"]); L = float(meta["length"]); Ma = float(meta["Ma"])
    msh = G.create_mesh(L, res); W = G.space_comp(msh)
    bcs = make_bcs(msh, W, meta, L)
    wx = build_X(msh, W, d, eps); wy = build_Y(msh, W, d)

    F = residual(msh, W, wx, Ra, eps, Ma)
    Lf = -ufl.replace(F, {wx: wx})
    J = ufl.derivative(F, wx)
    Jf = ufl.replace(J, {wx: wx})
    print("assembling ...")
    problem = fem.petsc.LinearProblem(
        Jf, Lf, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    delta = problem.solve()
    print("solved; ||delta||=%.3e" % np.linalg.norm(delta.x.array))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "论文", "direct")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "%s_e%.1f.npz" % (case, eps)),
             du=delta.sub(0).collapse().x.array.copy(),
             de=delta.sub(1).collapse().x.array.copy(),
             dr=delta.sub(2).collapse().x.array.copy(),
             dp=delta.sub(3).collapse().x.array.copy())
    print("delta norms u=%.3e e=%.3e rho=%.3e p=%.3e" %
          (np.linalg.norm(delta.sub(0).collapse().x.array),
           np.linalg.norm(delta.sub(1).collapse().x.array),
           np.linalg.norm(delta.sub(2).collapse().x.array),
           np.linalg.norm(delta.sub(3).collapse().x.array)))
    # compare channels
    def arr(f, i): return f.sub(i).collapse().x.array
    du = arr(delta, 0); de = arr(delta, 1); dp = arr(delta, 3)
    p_y = arr(wy, 3); p_x = arr(wx, 3)
    uy_x = arr(wx, 0); uy_y = arr(wy, 0)
    T_x = arr(wx, 1); T_y = arr(wy, 1)
    coords_p = load(d, "coords_p_bouss")[:, 1]
    Dp = deramp(p_y - p_x, coords_p)
    Pp = deramp(dp, coords_p)
    uxd = du.reshape(-1, 2); uyd = uy_y.reshape(-1, 2) - uy_x.reshape(-1, 2)
    ch = {
        "ux": (uxd[:, 0], uyd[:, 0]),
        "uy": (uxd[:, 1], uyd[:, 1]),
        "theta": (de, T_y - T_x),
        "p": (Pp, Dp),
    }
    for k, (pred, meas) in ch.items():
        e = np.linalg.norm(pred - meas) / np.linalg.norm(meas)
        print("%-5s ||pred||=%.3e ||meas||=%.3e  relerr=%.3f  corr=%.4f" %
              (k, np.linalg.norm(pred), np.linalg.norm(meas), e,
               np.corrcoef(pred, meas)[0, 1]))


if __name__ == "__main__":
    main()

