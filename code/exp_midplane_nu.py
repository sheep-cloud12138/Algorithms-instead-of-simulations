"""Validate a wall-stencil-free Nusselt estimator: the mid-plane enthalpy flux.

For the Boussinesq member the steady energy balance gives, at any vertical cut x0,
    (1/kappa) * Int_0^1 (u_x theta - kappa dtheta/dx) dy = Nu_wall,
with kappa = (Ra Pr)^-1/2, provided the top/bottom walls are adiabatic.
"""
import csv, json, sys
import numpy as np

sys.path.insert(0, r"D:\Desktop\算法代替仿真\code")
from pt_numpy import load

DATA = r"D:\Desktop\算法代替仿真\data_public"
DT, PR = 1.2, 0.71

X = load(DATA + r"\test_legacy_X.pt")[:, :3].astype(np.float64)
Y = load(DATA + r"\test_legacy_Y.pt")[:, :3].astype(np.float64)
rows = list(csv.DictReader(open(DATA + r"\test_legacy_per_sample.csv")))
Ra = np.array([float(r["Ra"]) for r in rows])

th_X = (X[:, 2] - (1 - DT / 2)) / DT
th_Y = (Y[:, 2] - (1 - DT / 2)) / DT
ux_X, ux_Y = X[:, 0], Y[:, 0]


def wall_nu(t, n=6):
    h = 1.0 / (t.shape[-1] - 1)
    w = np.linalg.solve(np.vander(np.arange(n) * h, n, increasing=True).T, np.eye(n)[1])
    return -(t[:, :, :n] * w).sum(-1)


def mid_nu(ux, t, ra, x0=None):
    kap = (1.0 / np.sqrt(ra * PR))[:, None]
    dx = 1.0 / (t.shape[-1] - 1)
    dy = 1.0 / (t.shape[-2] - 1)
    i = t.shape[-1] // 2 if x0 is None else x0
    t_x = (t[:, :, i + 1] - t[:, :, i - 1]) / (2 * dx)
    f = ux[:, :, i] * t[:, :, i] - kap * t_x
    return np.trapezoid(f, dx=dy, axis=1) / kap[:, 0]


for name, ux, t in (("X (Boussinesq)", ux_X, th_X), ("Y (compressible)", ux_Y, th_Y)):
    wn = wall_nu(t).mean(1)
    mn = mid_nu(ux, t, Ra)
    rel = np.abs(mn - wn) / wn
    print(f"{name:18s} wall={wn.mean():8.3f}  mid={mn.mean():8.3f}  "
          f"rel diff median={np.median(rel)*100:6.2f}%  mean={rel.mean()*100:6.2f}%  "
          f"max={rel.max()*100:6.2f}%")


