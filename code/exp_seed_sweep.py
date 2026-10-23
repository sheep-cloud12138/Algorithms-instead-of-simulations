"""Five-seed audit of the transfer, fair-baseline and wall-flux results.

Fixes the normalisation protocol: a model trained on the public 4000 pairs must
see target data scaled by the *source* per-channel std, not by the target's own
std.  All models use the same 150-epoch protocol as the paper.

Outputs
  ~/nc_run/p7/seed_sweep/*.pt          weights
  论文/seed_sweep.json                 every per-seed and aggregate number
"""
import argparse
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

PUB = os.path.expanduser("~/nc_run/pub")
P7 = os.path.expanduser("~/nc_run/p7")
OUT = os.path.join(os.path.dirname(HERE), "论文", "seed_sweep.json")
DTP, M = 1.2, 6
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def theta(T):
    return (T - (1.0 - DTP / 2.0)) / DTP


def wall_q(th, w):
    ql = -(th[:, :, :M] * w).sum(-1)
    qr = (th[:, :, -M:].flip(-1) * w).sum(-1)
    return ql, qr


def stencil(n, m=M):
    xw = np.arange(m) * (1.0 / (n - 1))
    return torch.tensor(np.linalg.solve(np.vander(xw, m, increasing=True).T, np.eye(m)[1]),
                        dtype=torch.float32)


def train_one(kind, seed, IN, target, epochs, lam, Xth, Dth, w, sD_mode, ckpt):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = UNet(4, 32).to(DEV)
    if sD_mode == "perchannel":
        sD = sd_of(target, len(target))
    else:
        sD = torch.tensor(float(target.std())).view(1, 1, 1, 1)
    Tn = target / sD
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx = np.arange(len(IN))
    bs = 32
    t0 = time.time()
    sdt = float(sD.flatten()[2]) if sD.numel() > 1 else float(sD)
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
                th = Xth[b].to(DEV) + DTP * out[:, 2] * sdt
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
            print("  %-28s ep %3d/%d %5.0fs" % (kind, ep + 1, epochs, time.time() - t0),
                  flush=True)
    path = os.path.join(ckpt, "%s_s%d.pt" % (kind, seed))
    torch.save({"sd": {k: v.cpu() for k, v in net.state_dict().items()},
                "sD": sD.cpu(), "seed": seed}, path)
    return path, time.time() - t0


def load_net(path):
    c = torch.load(path, weights_only=False)
    net = UNet(4, 32).to(DEV)
    net.load_state_dict(c["sd"])
    net.eval()
    return net, c["sD"].to(DEV)


def predict(net, IN, sD, idx):
    out = []
    with torch.no_grad():
        for k in range(0, len(idx), 64):
            b = idx[k:k + 64]
            out.append((net(IN[b].to(DEV)) * sD).cpu())
    return torch.cat(out)


def gerr(Yp, Yt):
    return [float((((Yp[:, c] - Yt[:, c]) ** 2).sum() / (Yt[:, c] ** 2).sum()) ** 0.5)
            for c in range(4)]


def per_sample(Yp, Yt, wc):
    t = torch.norm(Yp[:, 2] - Yt[:, 2], dim=(1, 2)) / torch.norm(Yt[:, 2], dim=(1, 2))
    qh = wall_q(Yp[:, 2], wc)
    qt = wall_q(Yt[:, 2], wc)
    hot = ((qh[0].mean(1) - qt[0].mean(1)).abs() / qt[0].mean(1).abs())
    cold = ((qh[1].mean(1) - qt[1].mean(1)).abs() / qt[1].mean(1).abs())
    both = ((qh[0].mean(1) - qt[0].mean(1)).abs() +
            (qh[1].mean(1) - qt[1].mean(1)).abs()) / \
           (qt[0].mean(1).abs() + qt[1].mean(1).abs())
    loc = ((qh[0] - qt[0]).abs().mean(1) / qt[0].abs().mean(1) +
           (qh[1] - qt[1]).abs().mean(1) / qt[1].abs().mean(1)) / 2
    return {"T": t.numpy(), "Nu_hot": hot.numpy(), "Nu_cold": cold.numpy(),
            "Nu_both": both.numpy(), "local": loc.numpy()}


def summarize(v):
    return {"mean": float(np.mean(v)), "median": float(np.median(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def boot_diff(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--sD", default="perchannel", choices=["perchannel", "scalar"])
    ap.add_argument("--own", action="store_true", help="also train within-family models")
    ap.add_argument("--own-seeds", type=int, default=5)
    ap.add_argument("--tag", default="seed_sweep")
    args = ap.parse_args()
    ckpt = os.path.join(P7, args.tag)
    os.makedirs(ckpt, exist_ok=True)

    Xp = torch.load(PUB + "/train/X.pt").float()
    Yp = torch.load(PUB + "/train/Y.pt").float()
    Xl = torch.load(PUB + "/test_legacy/X.pt").float()
    Yl = torch.load(PUB + "/test_legacy/Y.pt").float()
    Xp[:, 2], Yp[:, 2] = theta(Xp[:, 2]), theta(Yp[:, 2])
    Xl[:, 2], Yl[:, 2] = theta(Xl[:, 2]), theta(Yl[:, 2])
    sxp = Xp.reshape(len(Xp), 4, -1).std(2).mean(0).view(1, 4, 1, 1)
    INp = Xp / sxp
    INl = Xl / sxp
    Dp = Yp - Xp
    w = stencil(Xp.shape[-1]).to(DEV)

    X6, Y6, eps6, _ = load600()
    IN6 = X6 / sxp
    ev6 = torch.tensor(eps6).view(-1, 1, 1, 1)
    D6 = Y6 - X6

    res = {"protocol": vars(args), "source_std": sxp.flatten().tolist(), "seeds": {}}

    for seed in range(args.seeds):
        print("=== seed %d ===" % seed, flush=True)
        r = {}
        for kind, target, lam in [
            ("pub_plain", Dp, 0.0),
            ("pub_prior", Dp / DTP, 0.0),
            ("pub_flux", Dp / DTP, args.lam),
        ]:
            path = os.path.join(ckpt, "%s_s%d.pt" % (kind, seed))
            if os.path.exists(path):
                print("  [cache] %s" % kind, flush=True)
            else:
                train_one(kind, seed, INp, target, args.epochs, lam,
                          Xp[:, 2], Dp[:, 2], w, args.sD, ckpt)
            net, sD = load_net(path)
            r[kind] = {}
            for split in ("extrap", "interp"):
                tr, te = splits(eps6, split)
                idx = np.where(te)[0]
                d = predict(net, IN6, sD, idx)
                if kind == "pub_plain":
                    pred = {"plain": X6[idx] + d,
                            "plain_rescaled": X6[idx] + (ev6[idx] / DTP) * d}
                else:
                    pred = {"prior": X6[idx] + ev6[idx] * d}
                r[kind][split] = {"n": int(len(idx))}
                for tag, Yh in pred.items():
                    r[kind][split][tag] = gerr(Yh, Y6[idx])
            # per-sample wall statistics on the classical-cavity split
            idx = np.arange(len(Xl))
            d = predict(net, INl, sD, idx)
            Yh = Xl + DTP * d if kind != "pub_plain" else Xl + d
            ps = per_sample(Yh, Yl, w.cpu())
            r[kind]["legacy"] = {k: summarize(v) for k, v in ps.items()}
            t, nu = ps["T"], ps["Nu_both"]
            rho, pv = stats.spearmanr(t, nu)
            r[kind]["legacy"]["frac_T1_Nu2"] = float(((t < 0.01) & (nu > 0.02)).mean())
            r[kind]["legacy"]["spearman"] = float(rho)
            r[kind]["legacy"]["p"] = float(pv)
            r[kind]["legacy"]["_T"] = t.tolist()
            r[kind]["legacy"]["_Nu"] = nu.tolist()
            r[kind]["legacy"]["_local"] = ps["local"].tolist()
        res["seeds"][str(seed)] = r
        json.dump(res, open(OUT, "w"), indent=1)

    # aggregate the legacy per-sample arrays across seeds
    agg = {}
    for kind in ("pub_plain", "pub_prior", "pub_flux"):
        T = np.array([res["seeds"][str(s)][kind]["legacy"]["_T"]
                      for s in range(args.seeds)])
        N = np.array([res["seeds"][str(s)][kind]["legacy"]["_Nu"]
                      for s in range(args.seeds)])
        L = np.array([res["seeds"][str(s)][kind]["legacy"]["_local"]
                      for s in range(args.seeds)])
        agg[kind] = {
            "T_seed_mean": float(T.mean()), "T_seed_sd": float(T.mean(1).std(ddof=1)),
            "Nu_seed_mean": float(N.mean()), "Nu_seed_sd": float(N.mean(1).std(ddof=1)),
            "local_seed_mean": float(L.mean()), "local_seed_sd": float(L.mean(1).std(ddof=1)),
            "frac_T1_Nu2_mean": float(np.mean([((T[s] < 0.01) & (N[s] > 0.02)).mean()
                                               for s in range(args.seeds)])),
        }
    agg["prior_vs_flux_Nu_ci"] = boot_diff(
        np.array([res["seeds"][str(s)]["pub_prior"]["legacy"]["_Nu"]
                  for s in range(args.seeds)]).mean(0),
        np.array([res["seeds"][str(s)]["pub_flux"]["legacy"]["_Nu"]
                  for s in range(args.seeds)]).mean(0))
    agg["prior_vs_floor_Nu_ci"] = boot_diff(
        np.array([res["seeds"][str(s)]["pub_prior"]["legacy"]["_Nu"]
                  for s in range(args.seeds)]).mean(0),
        np.array([res["seeds"][str(s)]["pub_plain"]["legacy"]["_Nu"]
                  for s in range(args.seeds)]).mean(0))
    res["aggregate_legacy"] = agg

    # summary of transfer numbers
    tr = {}
    for split in ("extrap", "interp"):
        tr[split] = {}
        for tag in ("plain", "plain_rescaled", "prior"):
            vals = []
            for s in range(args.seeds):
                kind = "pub_plain" if tag.startswith("plain") else "pub_prior"
                vals.append(res["seeds"][str(s)][kind][split][tag])
            v = np.array(vals)
            tr[split][tag] = {"mean": v.mean(0).tolist(), "sd": v.std(0, ddof=1).tolist()}
    res["transfer"] = tr
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
