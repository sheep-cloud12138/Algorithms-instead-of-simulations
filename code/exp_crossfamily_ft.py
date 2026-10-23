"""Last-layer adaptation pre-run for narrow -> broad family transfer."""
import copy
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import gerr, load600, splits
from common import UNet

LEG = "/mnt/d/Desktop/算法代替仿真/research/data/train_legacy"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "后续方案", "prelim_crossfamily_ft.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_legacy(path):
    X = torch.load(os.path.join(path, "X.pt"), weights_only=False, mmap=True).float()
    rows = list(csv.DictReader(open(os.path.join(path, "per_sample.csv"))))
    dT = np.array([float(r["dT"]) for r in rows], np.float32)
    T0 = np.array([float(r["T0"]) for r in rows], np.float32)
    eps = dT / T0
    for i in range(len(X)):
        X[i, 2] = (X[i, 2] - (1.0 - dT[i] / (2.0 * T0[i]))) / eps[i]
    return X


def main():
    Xl = load_legacy(LEG)
    sx = Xl.reshape(len(Xl), 4, -1).std(2).mean(0).view(1, 4, 1, 1)
    cache = torch.load(os.path.expanduser("~/nc_run/p7/pub_legacy_nets.pt"),
                       weights_only=False)
    base = UNet(4, 32)
    base.load_state_dict(cache["legacy_naive"]["sd"])
    base = base.to(DEV)
    sD_cpu = cache["legacy_naive"]["sD"]
    sD = sD_cpu.to(DEV)

    X6, Y6, eps6, _ = load600()
    D6 = Y6 - X6
    ev6 = torch.tensor(eps6, dtype=torch.float32).view(-1, 1, 1, 1)
    cal = (eps6 >= 1.1) & (eps6 <= 1.3)
    tr6, te6 = splits(eps6, "extrap")
    pool = np.where(cal & tr6)[0]
    test = np.where(te6)[0]
    IN = (X6 / sx).to(DEV)
    Dt = D6.to(DEV)

    out = {"n_pool": int(len(pool)), "n_test": int(len(test)),
           "floor": gerr(X6[test], Y6[test]),
           "direct": gerr(X6[test] + ev6[test] * base(IN[test]).cpu() * sD_cpu,
                          Y6[test]), "runs": []}
    print("floor ", " ".join("%.4f" % v for v in out["floor"]))
    print("direct", " ".join("%.4f" % v for v in out["direct"]))
    rng = np.random.default_rng(20261019)
    for k in (5, 10, 20):
        for seed in range(5):
            idx = torch.tensor(rng.choice(pool, k, replace=False), device=DEV)
            net = copy.deepcopy(base)
            for p in net.parameters():
                p.requires_grad_(False)
            for p in net.o.parameters():
                p.requires_grad_(True)
            opt = torch.optim.Adam(net.o.parameters(), lr=1e-3, weight_decay=1e-4)
            target = Dt[idx] / sD
            for _ in range(150):
                pr = net(IN[idx])
                loss = ((pr - target) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                dp = net(IN[test]).cpu() * sD_cpu
            err = gerr(X6[test] + ev6[test] * dp, Y6[test])
            out["runs"].append({"k": k, "seed": seed, "err": err})
            print("k=%-2d s=%d " % (k, seed), " ".join("%.4f" % v for v in err))
    for k in (5, 10, 20):
        e = np.array([r["err"] for r in out["runs"] if r["k"] == k])
        out["k%d" % k] = {"median": np.median(e, 0).round(5).tolist(),
                          "p10": np.percentile(e, 10, axis=0).round(5).tolist(),
                          "p90": np.percentile(e, 90, axis=0).round(5).tolist()}
        print("k=%-2d median" % k, " ".join("%.4f" % v for v in out["k%d" % k]["median"]))
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
