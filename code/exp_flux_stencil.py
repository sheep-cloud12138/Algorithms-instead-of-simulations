"""Evaluate the wall-flux-trained prior with 4-, 6- and 8-point stencils."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import build_input
sys.path.insert(0, os.path.expanduser('~/nc_run/p4'))
from common import UNet

PUB = os.path.expanduser('~/nc_run/pub'); P7 = os.path.expanduser('~/nc_run/p7')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DTP = 1.2


def theta(T):
    return (T - (1.0 - DTP / 2.0)) / DTP


def wts(n, M):
    x = np.arange(M) * (1.0 / (n - 1))
    return np.linalg.solve(np.vander(x, M, increasing=True).T, np.eye(M)[1])


def q(th, M):
    th = th.detach().cpu().numpy() if hasattr(th, "detach") else th
    w = wts(th.shape[-1], M)
    ql = -(th[:, :, :M] * w).sum(-1)
    qr = (th[:, :, -M:][:, :, ::-1] * w).sum(-1)
    return ql, qr


Xte = torch.load(PUB + '/test_legacy/X.pt').float()
Yte = torch.load(PUB + '/test_legacy/Y.pt').float()
Xte[:, 2], Yte[:, 2] = theta(Xte[:, 2]), theta(Yte[:, 2])
Xtr = torch.load(PUB + '/train/X.pt').float()
Xtr[:, 2] = theta(Xtr[:, 2])
_, sx = build_input(Xtr, np.full(len(Xtr), DTP, np.float32), False)
IN = Xte / sx

out = {}
for tag, fn in (('base', None), ('lam1_s0', 'pub_flux_lam1.0.pt'), ('lam1_s1', 'pub_flux_lam1.0_s1.pt')):
    if fn is None:
        Dp = torch.load(P7 + '/pub_nets.pt', weights_only=False)['pub_prior']
        net = UNet(4, 32).to(DEV); net.load_state_dict(Dp['sd']); sD = Dp['sD']
    else:
        ck = torch.load(os.path.join(P7, fn), weights_only=False)
        net = UNet(4, 32).to(DEV); net.load_state_dict(ck['sd']); sD = ck['sD']
    net.eval(); ps = []
    with torch.no_grad():
        for k in range(0, len(IN), 64):
            ps.append(net(IN[k:k + 64].to(DEV)).cpu() * sD)
    Yh = Xte + DTP * torch.cat(ps)
    row = {}
    for M in (4, 6, 8):
        e = {}
        for wall, i in (('hot', 0), ('cold', 1)):
            qt = q(Yte[:, 2], M)[i]; qh = q(Yh[:, 2], M)[i]
            e[wall] = float((np.abs(qh.mean(1) - qt.mean(1)).mean() / np.abs(qt.mean(1)).mean()))
        e['floor'] = float(np.abs(q(Xte[:, 2], M)[0].mean(1) - q(Yte[:, 2], M)[0].mean(1)).mean()
                           / np.abs(q(Yte[:, 2], M)[0].mean(1)).mean())
        row['M%d' % M] = e
    row['relL2_T'] = float(torch.norm(Yh[:, 2] - Yte[:, 2]) / torch.norm(Yte[:, 2]))
    out[tag] = row
    print(tag, {k: (round(v, 5) if isinstance(v, float) else {kk: round(vv, 5) for kk, vv in v.items()})
                for k, v in row.items()}, flush=True)

json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '\u8bba\u6587', 'public_flux_stencil.json'), 'w'), indent=1)
