"""Wall-flux penalty against the mean of the 4/6/8/10-point estimators.

Tests whether an estimator-robust target removes the matched-stencil behaviour
seen when the penalty is tied to a single stencil order.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import build_input
sys.path.insert(0, os.path.expanduser('~/nc_run/p4'))
from common import UNet

PUB = os.path.expanduser('~/nc_run/pub')
P7 = os.path.expanduser('~/nc_run/p7')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '\u8bba\u6587')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DTP, LAM = 1.2, 1.0
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ORDERS = (4, 6, 8, 10)


def theta(T):
    return (T - (1.0 - DTP / 2.0)) / DTP


def weights(points, n):
    x = np.arange(points) * (1.0 / (n - 1))
    return np.linalg.solve(np.vander(x, points, increasing=True).T, np.eye(points)[1])


def flux_order(th, w):
    m = len(w)
    ql = -(th[:, :, :m] * w).sum(-1)
    qr = (th[:, :, -m:].flip(-1) * w).sum(-1)
    return ql, qr


def flux_multi(th, ws):
    q = [flux_order(th, w) for w in ws]
    return (sum(a for a, _ in q) / len(q), sum(b for _, b in q) / len(q))


Xtr = torch.load(PUB + '/train/X.pt').float()
Ytr = torch.load(PUB + '/train/Y.pt').float()
Xte = torch.load(PUB + '/test_legacy/X.pt').float()
Yte = torch.load(PUB + '/test_legacy/Y.pt').float()
Xtr[:, 2], Ytr[:, 2] = theta(Xtr[:, 2]), theta(Ytr[:, 2])
Xte[:, 2], Yte[:, 2] = theta(Xte[:, 2]), theta(Yte[:, 2])
D = (Ytr - Xtr) / DTP

IN, sx = build_input(Xtr, np.full(len(Xtr), DTP, np.float32), False)
INte = Xte / sx

torch.manual_seed(SEED); np.random.seed(SEED)
net = UNet(4, 32).to(DEV)
sD = float(D.std())
Tn = D / sD
WS = [torch.tensor(weights(m, Xte.shape[-1]), dtype=torch.float32, device=DEV) for m in ORDERS]
Xth_tr = Xtr[:, 2].to(DEV)
Dth_tr = D[:, 2].to(DEV)

opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 150)
idx = np.arange(len(IN)); bs = 32; t0 = time.time()
for ep in range(150):
    net.train(); np.random.shuffle(idx); rel_s = wall_s = 0.0; nb = 0
    for k in range(0, len(idx), bs):
        b = torch.from_numpy(idx[k:k + bs])
        out = net(IN[b].to(DEV))
        dt = Tn[b].to(DEV)
        rel = ((out - dt).pow(2).mean((1, 2, 3)) / (dt.pow(2).mean((1, 2, 3)) + 1e-12)).mean()
        th = Xth_tr[b] + DTP * out[:, 2] * sD
        th_true = Xth_tr[b] + Dth_tr[b]
        ql, qr = flux_multi(th, WS)
        tl, tr = flux_multi(th_true, WS)
        wall = ((ql - tl).abs().mean() + (qr - tr).abs().mean()) / (tl.abs().mean() + tr.abs().mean())
        loss = rel + LAM * wall
        opt.zero_grad(); loss.backward(); opt.step()
        rel_s += float(rel.detach()); wall_s += float(wall.detach()); nb += 1
    sch.step()
    if (ep + 1) % 25 == 0:
        print('ep %3d  rel %.5f  wall %.5f  %5.0fs' % (ep + 1, rel_s / nb, wall_s / nb, time.time() - t0), flush=True)

path = os.path.join(P7, 'pub_flux_blend_s%d.pt' % SEED)
torch.save({'sd': {k: v.cpu() for k, v in net.state_dict().items()}, 'sD': sD}, path)
print('saved', path)
