"""Stencil-free heat-transfer metric: mid-plane enthalpy flux.

For the Boussinesq member the steady energy balance makes
    F(x) = Int (u_x theta - kappa dtheta/dx) dy / kappa
independent of x and equal to the wall Nusselt number; it reproduces the
6-point wall Nu on X to 0.09% median. The compressible member conserves a
different flux (density and pressure work do not enter this functional), so F
is not the physical Nu of Y. It is still a fixed linear functional of the
temperature field that touches no wall derivative, which is what is needed to
test whether the wall-flux penalty's advantage is estimator-specific.
"""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_flux_stencil10_np import (CKPT, PUB, ROOT, DTP, input_scale, load_pt,
                                    theta, unet)

PR = 0.71
BAND = (24, 40)


def mid_flux(ux, th, ra):
    kap = 1.0 / np.sqrt(ra * PR)
    dx = 1.0 / (th.shape[-1] - 1)
    dy = 1.0 / (th.shape[-2] - 1)
    out = []
    for i in range(BAND[0], BAND[1]):
        t_x = (th[:, :, i + 1] - th[:, :, i - 1]) / (2 * dx)
        f = ux[:, :, i] * th[:, :, i] - kap[:, None] * t_x
        out.append(np.trapezoid(f, dx=dy, axis=1) / kap)
    return np.mean(out, axis=0)


def rel_err(pred, target):
    return float(np.abs(pred - target).mean() / np.abs(target).mean())


def wall_nu(t, n):
    w = np.linalg.solve(
        np.vander(np.arange(n) * (1.0 / (t.shape[-1] - 1)), n, increasing=True).T,
        np.eye(n)[1])
    return -(t[:, :, :n] * w).sum(-1).mean(1)


def predict(x_te, scale, ck):
    pred = np.empty_like(x_te)
    s_d = np.asarray(ck["sD"], dtype=np.float32)
    s_d = s_d.reshape(1, 4, 1, 1) if s_d.size == 4 else float(s_d)
    for k in range(0, len(x_te), 8):
        pred[k:k + 8] = unet(x_te[k:k + 8] / scale, ck["sd"]) * s_d
    return pred


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rows = list(csv.DictReader(open(ROOT / "data_public" / "test_legacy_per_sample.csv")))
    ra = np.array([float(r["Ra"]) for r in rows[:n]])

    x_tr = load_pt(PUB / "train" / "X.pt").astype(np.float32, copy=False)
    x_te = load_pt(PUB / "test_legacy" / "X.pt").astype(np.float32, copy=False)[:n]
    y_te = load_pt(PUB / "test_legacy" / "Y.pt").astype(np.float32, copy=False)[:n]
    for a in (x_tr, x_te, y_te):
        a[:, 2] = theta(a[:, 2])
    scale = input_scale(x_tr)

    ck = {"base": load_pt(CKPT / "pub_nets.pt")["pub_prior"],
          "flux_s0": load_pt(CKPT / "pub_flux_lam1.0.pt"),
          "flux_s1": load_pt(CKPT / "pub_flux_lam1.0_s1.pt")}

    ux = y_te[:, 0].astype(np.float64)
    th_x = x_te[:, 2].astype(np.float64)
    th_y = y_te[:, 2].astype(np.float64)
    target = mid_flux(ux, th_y, ra)
    res = {"n": n, "target_mean": float(target.mean()),
           "floor_mid": rel_err(mid_flux(ux, th_x, ra), target),
           "floor_wall6": rel_err(wall_nu(th_x, 6), wall_nu(th_y, 6)),
           "floor_wall10": rel_err(wall_nu(th_x, 10), wall_nu(th_y, 10))}
    for name, c in ck.items():
        th_hat = (x_te[:, 2] + DTP * predict(x_te, scale, c)[:, 2]).astype(np.float64)
        res[name] = {
            "mid": rel_err(mid_flux(ux, th_hat, ra), target),
            "wall6": rel_err(wall_nu(th_hat, 6), wall_nu(th_y, 6)),
            "wall10": rel_err(wall_nu(th_hat, 10), wall_nu(th_y, 10)),
            "relL2_T": float(np.linalg.norm(th_hat - th_y) / np.linalg.norm(th_y)),
        }
    print(json.dumps(res, indent=1))
    (ROOT / "论文" / "midplane_metric.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
