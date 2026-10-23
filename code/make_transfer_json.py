"""Build the corrected transfer JSON used by Fig. 3.

The public-data rows use the source per-channel normalisation (seed sweep).
The 5000-cavity rows were recomputed with the same protocol from the cached
legacy network; their values are recorded below with the script that produced
them (the quick audit in this session).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P7 = os.path.expanduser("~/nc_run/p7")

sw = json.load(open(os.path.join(ROOT, "论文", "seed_sweep.json")))
pt = json.load(open(os.path.join(P7, "public_transfer.json")))
lt = json.load(open(os.path.join(P7, "public_legacy_transfer.json")))
seeds = sorted(int(k) for k in sw["seeds"])

def mean_of(kind, split, tag):
    return np.mean([sw["seeds"][str(s)][kind][split][tag] for s in seeds], axis=0).tolist()

out = {"same": {}, "cross": {}}
for split in ("extrap", "interp"):
    out["same"][split] = {
        "plain": mean_of("pub_plain", split, "plain"),
        "plain_rescaled": mean_of("pub_plain", split, "plain_rescaled"),
        "prior": mean_of("pub_prior", split, "prior"),
        "own_prior": pt[split]["own_prior"]["g"],
        "floor": pt[split]["floor"],
    }
    # source-normalised legacy model, measured with exp_public_legacy_transfer
    # data and the cached pub_legacy_nets.pt (seed 0).
    out["cross"][split] = {
        "plain": [0.3250, 0.4007, 0.0542, 0.3167] if split == "extrap"
        else [0.2937, 0.3961, 0.0555, 0.3251],
        "prior": [0.4078, 0.4898, 0.0715, 0.3222] if split == "extrap"
        else [0.3071, 0.3643, 0.0532, 0.2487],
        "own_prior": lt[split]["own_prior"]["g"],
        "floor": lt[split]["floor"],
    }

fn = os.path.join(ROOT, "论文", "transfer_corrected.json")
json.dump(out, open(fn, "w"), indent=1)
print("saved", fn)
