"""Both-wall Nu error at 4/6/8/10 points for arbitrary recovered checkpoints."""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_flux_stencil10_np import CKPT, PUB, DTP, input_scale, load_pt, theta, unet, wall_flux


def both_wall(th_hat, th_true, points):
    t = wall_flux(th_true, points)
    p = wall_flux(th_hat, points)
    f = wall_flux(th_true * 0 + th_true, points)
    err = []
    for i in (0, 1):
        err.append(np.abs(p[i].mean(1) - t[i].mean(1)).mean() / np.abs(t[i].mean(1)).mean())
    return float(np.mean(err))


def main():
    n = 200
    x_tr = load_pt(PUB / "train" / "X.pt").astype(np.float32, copy=False)
    x_te = load_pt(PUB / "test_legacy" / "X.pt").astype(np.float32, copy=False)[:n]
    y_te = load_pt(PUB / "test_legacy" / "Y.pt").astype(np.float32, copy=False)[:n]
    for a in (x_tr, x_te, y_te):
        a[:, 2] = theta(a[:, 2])
    scale = input_scale(x_tr)

    names = sys.argv[1:]
    out = {}
    for name in names:
        ck = load_pt(CKPT / name)
        s_d = np.asarray(ck["sD"], dtype=np.float32)
        s_d = s_d.reshape(1, 4, 1, 1) if s_d.size == 4 else float(s_d)
        pred = np.empty_like(x_te)
        for k in range(0, n, 8):
            pred[k:k + 8] = unet(x_te[k:k + 8] / scale, ck["sd"]) * s_d
        th_hat = (x_te[:, 2] + DTP * pred[:, 2]).astype(np.float64)
        row = {f"{p}pt": round(100 * both_wall(th_hat, y_te[:, 2].astype(np.float64), p), 3)
               for p in (4, 6, 8, 10)}
        row["relL2_T"] = round(float(np.linalg.norm(th_hat - y_te[:, 2]) / np.linalg.norm(y_te[:, 2])) * 100, 4)
        out[name] = row
        print(name, row, flush=True)
    (Path(__file__).resolve().parents[1] / "论文" / "stencil_ckpt_sweep.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
