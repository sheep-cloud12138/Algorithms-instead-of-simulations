"""Per-sample field error vs integral-Nu error on the held-out cavity split."""
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import build_input
from common import UNet

PUB = os.path.expanduser("~/nc_run/pub")
P7 = os.path.expanduser("~/nc_run/p7")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "后续方案")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTP, M = 1.2, 6


def theta(T):
    return (T - (1.0 - DTP / 2.0)) / DTP


def wall_q(th, w):
    ql = -(th[:, :, :M] * w).sum(-1)
    qr = (th[:, :, -M:].flip(-1) * w).sum(-1)
    return ql, qr


def load_net(sd):
    net = UNet(4, 32)
    net.load_state_dict(sd)
    return net.to(DEV).eval()


def predict(net, sD, IN):
    with torch.no_grad():
        return torch.cat([net(IN[i:i + 64].to(DEV)).cpu() * sD
                          for i in range(0, len(IN), 64)])


def main():
    Xtr = torch.load(PUB + "/train/X.pt").float()
    Xte = torch.load(PUB + "/test_legacy/X.pt").float()
    Yte = torch.load(PUB + "/test_legacy/Y.pt").float()
    Xtr[:, 2], Xte[:, 2], Yte[:, 2] = theta(Xtr[:, 2]), theta(Xte[:, 2]), theta(Yte[:, 2])
    IN, sx = build_input(Xtr, np.full(len(Xtr), DTP, np.float32), False)
    INte = Xte / sx

    prior = torch.load(os.path.join(P7, "pub_nets.pt"), weights_only=False)["pub_prior"]
    flux = torch.load(os.path.join(P7, "pub_flux_lam1.0.pt"), weights_only=False)
    preds = {
        "floor": Xte,
        "prior": Xte + DTP * predict(load_net(prior["sd"]), prior["sD"], INte),
        "prior+flux": Xte + DTP * predict(load_net(flux["sd"]), flux["sD"], INte),
    }

    xw = np.arange(M) * (1.0 / (Xte.shape[-1] - 1))
    w = torch.tensor(np.linalg.solve(np.vander(xw, M, increasing=True).T, np.eye(M)[1]),
                     dtype=torch.float32)
    q_true = wall_q(Yte[:, 2], w)
    rows = []
    for i in range(len(Yte)):
        rec = {"i": i}
        for tag, Y in preds.items():
            rec[tag + "_T"] = float(torch.norm(Y[i, 2] - Yte[i, 2]) / torch.norm(Yte[i, 2]))
            qh = wall_q(Y[i:i + 1, 2], w)
            den = q_true[0][i].mean().abs() + q_true[1][i].mean().abs()
            num = (qh[0].mean() - q_true[0][i].mean()).abs() + \
                  (qh[1].mean() - q_true[1][i].mean()).abs()
            rec[tag + "_Nu"] = float(num / den)
        rows.append(rec)

    out = {"n": len(rows), "rows": rows}
    for tag in preds:
        t = np.array([r[tag + "_T"] for r in rows])
        n = np.array([r[tag + "_Nu"] for r in rows])
        rho, p = stats.spearmanr(t, n)
        out[tag] = {
            "T_median": float(np.median(t)), "T_mean": float(t.mean()),
            "Nu_median": float(np.median(n)), "Nu_mean": float(n.mean()),
            "spearman": float(rho), "p": float(p),
            "frac_T_below_1pct": float((t < 0.01).mean()),
            "frac_T_below_1pct_Nu_above_2pct": float(((t < 0.01) & (n > 0.02)).mean()),
        }
        print(tag, json.dumps(out[tag], indent=1))
    json.dump(out, open(os.path.join(OUT, "prelim_field_vs_nu.json"), "w"), indent=1)
    print("saved", os.path.join(OUT, "prelim_field_vs_nu.json"))


if __name__ == "__main__":
    main()
