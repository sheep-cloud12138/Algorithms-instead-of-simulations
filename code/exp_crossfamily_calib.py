"""Cross-family scalar calibration pre-run.

Train-free transfer: a public random-wall model (eps=1.2) is applied to the
classical-cavity family.  With k paired cavity samples at the same contrast,
fit one global (or channel-wise) amplitude on the predicted correction, then
evaluate on the 200 held-out cavity samples.  This is the cheapest possible
test of whether "one contrast, a few pairs" can rescue cross-family transfer.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import gerr
from common import UNet
from lib import load600, splits

PUB = os.path.expanduser("~/nc_run/pub")
LEG = "/mnt/d/Desktop/算法代替仿真/research/data"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "后续方案", "prelim_crossfamily_calib.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTP = 1.2


def load_split(path):
    X = torch.load(os.path.join(path, "X.pt"), weights_only=False, mmap=True).float()
    Y = torch.load(os.path.join(path, "Y.pt"), weights_only=False, mmap=True).float()
    rows = list(csv.DictReader(open(os.path.join(path, "per_sample.csv"))))
    dT = np.array([float(r["dT"]) for r in rows], np.float32)
    T0 = np.array([float(r["T0"]) for r in rows], np.float32)
    eps = dT / T0
    for i in range(len(X)):
        shift = 1.0 - dT[i] / (2.0 * T0[i])
        X[i, 2] = (X[i, 2] - shift) / eps[i]
        Y[i, 2] = (Y[i, 2] - shift) / eps[i]
    return X, Y, eps


def relerr(pred, true):
    return gerr(pred, true)


def fit_alpha(dp, dt, per_channel):
    """Least-squares amplitude(s) for dp ~ alpha * dt."""
    if per_channel:
        return np.array([
            float((dp[:, c] * dt[:, c]).sum() / (dp[:, c] * dp[:, c]).sum())
            for c in range(4)
        ])
    return np.array([float((dp * dt).sum() / (dp * dp).sum())])


def apply_alpha(dp, alpha):
    if alpha.size == 1:
        return alpha[0] * dp
    return dp * torch.tensor(alpha, dtype=dp.dtype).view(1, 4, 1, 1)


def main():
    cache = torch.load(os.path.join(os.path.expanduser("~/nc_run/p7"), "pub_nets.pt"),
                       weights_only=False)
    net = UNet(4, 32)
    net.load_state_dict(cache["pub_naive"]["sd"])
    net = net.to(DEV).eval()
    sD = cache["pub_naive"]["sD"]

    Xp = torch.load(os.path.join(PUB, "train", "X.pt"), weights_only=False,
                    mmap=True).float()
    sx = Xp.reshape(len(Xp), 4, -1).std(2).mean(0).view(1, 4, 1, 1)
    del Xp

    Xtr, Ytr, _ = load_split(os.path.join(LEG, "train_legacy"))
    Xte, Yte, _ = load_split(os.path.join(LEG, "test_legacy"))
    INtr = Xtr / sx
    INte = Xte / sx
    with torch.no_grad():
        Dtr = torch.cat([net(INtr[i:i + 64].to(DEV)).cpu() * sD
                         for i in range(0, len(INtr), 64)])
        Dte = torch.cat([net(INte[i:i + 64].to(DEV)).cpu() * sD
                         for i in range(0, len(INte), 64)])
    Ttr, Tte = Ytr - Xtr, Yte - Xte

    out = {"n_train": len(Xtr), "n_test": len(Xte),
           "floor": relerr(Xte, Yte), "direct_alpha1": relerr(Xte + Dte, Yte)}
    print("floor      ", " ".join("%.4f" % v for v in out["floor"]))
    print("direct a=1 ", " ".join("%.4f" % v for v in out["direct_alpha1"]))

    rng = np.random.default_rng(20261019)
    out["runs"] = []
    for k in (5, 10, 20, 50, 100):
        for mode in ("global", "channel"):
            errs = []
            alphas = []
            for _ in range(20):
                idx = rng.choice(len(Xtr), k, replace=False)
                alpha = fit_alpha(Dtr[idx].numpy(), Ttr[idx].numpy(),
                                  per_channel=(mode == "channel"))
                pred = Xte + apply_alpha(Dte, alpha)
                errs.append(relerr(pred, Yte))
                alphas.append(alpha.tolist())
            errs = np.array(errs)
            rec = {"k": k, "mode": mode,
                   "median": np.median(errs, 0).round(5).tolist(),
                   "p10": np.percentile(errs, 10, axis=0).round(5).tolist(),
                   "p90": np.percentile(errs, 90, axis=0).round(5).tolist(),
                   "alpha_median": np.median(np.array(alphas), 0).round(4).tolist()}
            out["runs"].append(rec)
            print("k=%-3d %-7s " % (k, mode),
                  " ".join("%.4f" % v for v in rec["median"]),
                  " alpha", rec["alpha_median"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)

    # Narrow -> broad: a classical-cavity model applied to random-wall data.
    cache2 = torch.load(os.path.join(os.path.expanduser("~/nc_run/p7"),
                                     "pub_legacy_nets.pt"), weights_only=False)
    net2 = UNet(4, 32)
    net2.load_state_dict(cache2["legacy_naive"]["sd"])
    net2 = net2.to(DEV).eval()
    sD2 = cache2["legacy_naive"]["sD"]
    sx2 = Xtr.reshape(len(Xtr), 4, -1).std(2).mean(0).view(1, 4, 1, 1)
    X6, Y6, eps6, _ = load600()
    D6 = Y6 - X6
    with torch.no_grad():
        D6p = torch.cat([net2((X6[i:i + 64] / sx2).to(DEV)).cpu() * sD2
                         for i in range(0, len(X6), 64)])
    ev6 = torch.tensor(eps6, dtype=torch.float32).view(-1, 1, 1, 1)
    cal = (eps6 >= 1.1) & (eps6 <= 1.3)
    tr6, te6 = splits(eps6, "extrap")
    idx_pool = np.where(cal & tr6)[0]
    idx_test = np.where(te6)[0]
    rec2 = {"n_pool": int(len(idx_pool)), "n_test": int(len(idx_test)),
            "floor": relerr(X6[idx_test], Y6[idx_test]),
            "direct_alpha1": relerr(X6[idx_test] + ev6[idx_test] * D6p[idx_test],
                                    Y6[idx_test])}
    print("\nnarrow->broad")
    print("floor      ", " ".join("%.4f" % v for v in rec2["floor"]))
    print("direct a=1 ", " ".join("%.4f" % v for v in rec2["direct_alpha1"]))
    rec2["runs"] = []
    rng = np.random.default_rng(20261019)
    for k in (5, 10, 20, 50, 100):
        if k > len(idx_pool):
            continue
        for mode in ("global", "channel"):
            errs, alphas = [], []
            for _ in range(20):
                idx = rng.choice(idx_pool, k, replace=False)
                alpha = fit_alpha(D6p[idx].numpy(), D6[idx].numpy(),
                                  per_channel=(mode == "channel"))
                pred = X6[idx_test] + ev6[idx_test] * apply_alpha(D6p[idx_test], alpha)
                errs.append(relerr(pred, Y6[idx_test]))
                alphas.append(alpha.tolist())
            errs = np.array(errs)
            r = {"k": k, "mode": mode,
                 "median": np.median(errs, 0).round(5).tolist(),
                 "p10": np.percentile(errs, 10, axis=0).round(5).tolist(),
                 "p90": np.percentile(errs, 90, axis=0).round(5).tolist(),
                 "alpha_median": np.median(np.array(alphas), 0).round(4).tolist()}
            rec2["runs"].append(r)
            print("k=%-3d %-7s " % (k, mode),
                  " ".join("%.4f" % v for v in r["median"]),
                  " alpha", r["alpha_median"])
    out["narrow_to_broad"] = rec2
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
