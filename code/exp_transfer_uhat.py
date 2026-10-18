"""Transfer criterion: the target's peak Boussinesq speed.

Evaluate the public random-wall prior on the public held-out splits and
correlate the per-sample channel-averaged error with
uhat = max sqrt(u_x^2 + u_y^2) of the Boussinesq input.
"""
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

PUB = os.path.expanduser('~/nc_run/pub')
CK = os.path.expanduser('~/nc_run/p7/pub_nets.pt')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   '论文', 'transfer_uhat.json')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 1.2
SPLITS = ['train', 'test_in_distribution', 'test_ood', 'test_legacy']


def load_split(name):
    X = torch.load(os.path.join(PUB, name, 'X.pt')).float()
    Y = torch.load(os.path.join(PUB, name, 'Y.pt')).float()
    rows = list(csv.DictReader(open(os.path.join(PUB, name, 'per_sample.csv'))))
    logRa = np.array([float(r['log10Ra']) for r in rows])
    return X, Y, logRa


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    rx = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ry = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(rx, ry)[0]


def main():
    Xtr, _, _ = load_split('train')
    _, sx = build_input(Xtr, np.full(len(Xtr), EPS, np.float32), False)
    cache = torch.load(CK, weights_only=False)['pub_prior']
    net = UNet(4, 32).to(DEV)
    net.load_state_dict(cache['sd'])
    net.eval()
    sD = cache['sD']

    rows, summary = [], {}
    for name in SPLITS:
        X, Y, logRa = load_split(name)
        IN = X / sx
        preds = []
        with torch.no_grad():
            for k in range(0, len(X), 64):
                preds.append(net(IN[k:k + 64].to(DEV)).cpu() * sD)
        P = torch.cat(preds)
        Yh = X + EPS * P
        num = ((Yh - Y) ** 2).sum((2, 3))
        den = (Y ** 2).sum((2, 3)) + 1e-12
        err = torch.sqrt(num / den).mean(1).numpy()
        uhat = np.sqrt(X[:, 0].numpy() ** 2 + X[:, 1].numpy() ** 2).max((1, 2))
        summary[name] = dict(n=int(len(X)), err_mean=float(err.mean()),
                             uhat_median=float(np.median(uhat)),
                             uhat_p10=float(np.percentile(uhat, 10)),
                             uhat_p90=float(np.percentile(uhat, 90)))
        for i in range(len(X)):
            rows.append(dict(split=name, err=float(err[i]), uhat=float(uhat[i]),
                             logRa=float(logRa[i])))
        print('%-22s n=%4d  err %.4f  uhat %.4f' %
              (name, len(X), err.mean(), np.median(uhat)), flush=True)

    e = np.array([r['err'] for r in rows])
    uh = np.array([r['uhat'] for r in rows])
    z = np.array([r['logRa'] for r in rows])
    rho = stats.spearmanr(uh, e)[0]
    prho = partial_spearman(uh, e, z)
    print('Spearman(uhat, err) %.3f  partial(logRa) %.3f' % (rho, prho))

    q = np.quantile(uh, [0, .2, .4, .6, .8, 1.0])
    bins = []
    for i in range(5):
        m = (uh >= q[i]) & (uh <= q[i + 1])
        bins.append(dict(lo=float(q[i]), hi=float(q[i + 1]), n=int(m.sum()),
                         err=float(e[m].mean())))
        print('uhat %.4f-%.4f  n=%4d  err %.4f' %
              (q[i], q[i + 1], m.sum(), e[m].mean()))

    json.dump(dict(splits=summary, spearman=float(rho), partial_logRa=float(prho),
                   bins=bins), open(OUT, 'w'), indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
