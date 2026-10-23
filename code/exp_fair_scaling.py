"""Fair baseline: take the plain model trained at eps0=1.2 and rescale its
predicted correction by eps/eps0 before adding it to X.

This is the baseline the paper must compare against: it uses exactly the same
network and data as the plain model, and only applies the known epsilon scaling
outside the network. If the prior is only a reparameterisation, this baseline
should match it.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import build_input, load600, predict_net, report, splits
from common import UNet

P7 = os.path.expanduser("~/nc_run/p7")
OUT = os.path.join(os.path.dirname(HERE), "论文", "fair_scaling.json")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS0 = 1.2


def load_net(cache, key):
    net = UNet(4, 32).to(DEV)
    net.load_state_dict(cache[key]["sd"])
    net.eval()
    return net, cache[key]["sD"]


X, Y, eps, _ = load600()
IN, _ = build_input(X, eps, False)
ev = torch.tensor(eps, dtype=torch.float32).view(-1, 1, 1, 1)

cache = torch.load(os.path.join(P7, "pub_nets.pt"), weights_only=False)
net_plain, sd_plain = load_net(cache, "pub_naive")
net_prior, sd_prior = load_net(cache, "pub_prior")

result = {}
for split in ("extrap", "interp"):
    tr, te = splits(eps, split)
    d_plain = predict_net(net_plain, IN, None, sd_plain, te, DEV)
    d_prior = predict_net(net_prior, IN, None, sd_prior, te, DEV)
    y_floor = X[te]
    y_plain = X[te] + d_plain
    y_scaled = X[te] + (ev[te] / EPS0) * d_plain
    y_prior = X[te] + ev[te] * d_prior
    variants = {
        "floor": y_floor,
        "plain": y_plain,
        "plain_rescaled": y_scaled,
        "prior": y_prior,
    }
    result[split] = {"n_test": int(te.sum())}
    for name, pred in variants.items():
        r = report(f"{split} {name}", pred, Y[te], y_floor)
        result[split][name] = {"g": r["g"]}

json.dump(result, open(OUT, "w"), indent=1)
print("saved", OUT)
