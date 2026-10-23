"""Within-family replication of the field-error / Nu-error decoupling.

Trains the prior and the wall-flux-constrained prior on our own 600 random-wall
pairs, five seeds, and reports per-sample field and integral-Nu errors on the
held-out interpolation split (including the eps ~ 1.2 subset).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import load600, splits, sd_of
from common import UNet

P7 = os.path.expanduser("~/nc_run/p7")
CKPT = os.path.join(P7, "own_seed_sweep")
OUT = os.path.join(os.path.dirname(HERE), "论文", "own_seed_sweep.json")
M = 6
DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CKPT, exist_ok=True)


def wall_q(th, w):
    ql = -(th[:, :, :M] * w).sum(-1)
    qr = (th[:, :, -M:].flip(-1) * w).sum(-1)
    return ql, qr


def stencil(n, m=M):
    xw = np.arange(m) * (1.0 / (n - 1))
    return torch.tensor(np.linalg.solve(np.vander(xw, m, increasing=True).T, np.eye(m)[1]),
                        dtype=torch.float32)


def gerr(Yp, Yt):
    return [float((((Yp[:, c] - Yt[:, c]) ** 2).sum() / (Yt[:, c] ** 2).sum()) ** 0.5)
            for c in range(4)]


def per_sample(Yp, Yt, wc):
    t = torch.norm(Yp[:, 2] - Yt[:, 2], dim=(1, 2)) / torch.norm(Yt[:, 2], dim=(1, 2))
    qh, qt = wall_q(Yp[:, 2], wc), wall_q(Yt[:, 2], wc)
    hot = (qh[0].mean(1) - qt[0].mean(1)).abs() / qt[0].mean(1).abs()
    cold = (qh[1].mean(1) - qt[1].mean(1)).abs() / qt[1].mean(1).abs()
    both = (hot * qt[0].mean(1).abs() + cold * qt[1].mean(1).abs()) / \
           (qt[0].mean(1).abs() + qt[1].mean(1).abs())
    loc = ((qh[0] - qt[0]).abs().mean(1) / qt[0].abs().mean(1) +
           (qh[1] - qt[1]).abs().mean(1) / qt[1].abs().mean(1)) / 2
    return {"T": t.numpy(), "Nu_hot": hot.numpy(), "Nu_cold": cold.numpy(),
            "Nu_both": both.numpy(), "local": loc.numpy()}


def train(kind, seed, IN, target, ev, Xth, Dth, w, lam, epochs=150):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = UNet(4, 32).to(DEV)
    sD = sd_of(target, len(target))
    Tn = target / sD
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx = np.arange(len(IN))
    bs, t0 = 16, time.time()
    sdt = float(sD.flatten()[2])
    for ep in range(epochs):
        net.train()
        np.random.shuffle(idx)
        for k in range(0, len(idx), bs):
            b = torch.from_numpy(idx[k:k + bs])
            out = net(IN[b].to(DEV))
            dt = Tn[b].to(DEV)
            rel = ((out - dt).pow(2).mean((1, 2, 3)) /
                   (dt.pow(2).mean((1, 2, 3)) + 1e-12)).mean()
            if lam > 0:
                th = Xth[b].to(DEV) + ev[b].to(DEV).view(-1, 1, 1) * out[:, 2] * sdt
                th_true = Xth[b].to(DEV) + Dth[b].to(DEV)
                ql, qr = wall_q(th, w)
                tl, tr = wall_q(th_true, w)
                wall = ((ql - tl).abs().mean() + (qr - tr).abs().mean()) / \
                       (tl.abs().mean() + tr.abs().mean())
                loss = rel + lam * wall
            else:
                loss = rel
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        if (ep + 1) % 50 == 0:
            print("  %-12s s%d ep %3d %5.0fs" % (kind, seed, ep + 1, time.time() - t0),
                  flush=True)
    path = os.path.join(CKPT, "%s_s%d.pt" % (kind, seed))
    torch.save({"sd": {k: v.cpu() for k, v in net.state_dict().items()},
                "sD": sD.cpu()}, path)
    return path


def main():
    X, Y, eps, _ = load600()
    ev = torch.tensor(eps).view(-1, 1, 1, 1)
    D = Y - X
    sx = X.reshape(len(X), 4, -1).std(2).mean(0).view(1, 4, 1, 1)
    IN = X / sx
    w = stencil(X.shape[-1]).to(DEV)
    res = {"seeds": {}}
    for seed in range(5):
        print("=== own seed %d ===" % seed, flush=True)
        r = {}
        for kind, lam in (("own_prior", 0.0), ("own_flux", 1.0)):
            path = os.path.join(CKPT, "%s_s%d.pt" % (kind, seed))
            tr, te = splits(eps, "interp")
            idx = np.where(tr)[0]
            if not os.path.exists(path):
                train(kind, seed, IN[tr], (D / ev)[tr], ev[tr], X[tr, 2], D[tr, 2],
                      w, lam)
            c = torch.load(path, weights_only=False)
            net = UNet(4, 32).to(DEV)
            net.load_state_dict(c["sd"])
            net.eval()
            out = []
            with torch.no_grad():
                for k in range(0, len(te), 64):
                    b = np.where(te)[0][k:k + 64]
                    out.append((net(IN[b].to(DEV)) * c["sD"].to(DEV)).cpu())
            d = torch.cat(out)
            Yh = X[te] + ev[te] * d
            r[kind] = {"g": gerr(Yh, Y[te])}
            ps = per_sample(Yh, Y[te], w.cpu())
            e = eps[te]
            r[kind]["all"] = {k: {"median": float(np.median(v)), "mean": float(np.mean(v))}
                              for k, v in ps.items()}
            m12 = (e >= 1.1) & (e <= 1.3)
            r[kind]["eps12"] = {k: {"median": float(np.median(v[m12])),
                                    "mean": float(np.mean(v[m12]))}
                                for k, v in ps.items()}
            t, nu = ps["T"][m12], ps["Nu_both"][m12]
            rho, pv = stats.spearmanr(t, nu)
            r[kind]["eps12"]["frac_T1_Nu2"] = float(((t < 0.01) & (nu > 0.02)).mean())
            r[kind]["eps12"]["spearman"] = float(rho)
            r[kind]["eps12"]["p"] = float(pv)
            r[kind]["n_eps12"] = int(m12.sum())
        res["seeds"][str(seed)] = r
        json.dump(res, open(OUT, "w"), indent=1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
