"""Public-data prior with a wall-flux term in the loss (derivative-aware training).

Same protocol as exp_public_transfer.py: 4000 public training pairs, eps=1.2, UNet(4,32),
150 epochs, seed 0, target D/eps.  The loss adds a relative L1 penalty on the wall-normal
temperature gradient at both side walls, evaluated with the same M-point stencil that
exp_public_nu.py uses for the Nusselt number.
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
DTP, LAM = 1.2, float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
M = int(sys.argv[3]) if len(sys.argv) > 3 else 6


def theta(T):
    return (T - (1.0 - DTP / 2.0)) / DTP


def wall_flux(th, w):
    ql = -(th[:, :, :M] * w).sum(-1)
    qr = (th[:, :, -M:].flip(-1) * w).sum(-1)
    return ql, qr


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
xw = np.arange(M) * (1.0 / (Xte.shape[-1] - 1))
w = torch.tensor(np.linalg.solve(np.vander(xw, M, increasing=True).T, np.eye(M)[1]),
                 dtype=torch.float32, device=DEV)
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
        ql, qr = wall_flux(th, w)
        tl, tr = wall_flux(th_true, w)
        if LAM > 0:
            wall = ((ql - tl).abs().mean() + (qr - tr).abs().mean()) / (tl.abs().mean() + tr.abs().mean())
            loss = rel + LAM * wall
        else:
            wall = torch.zeros((), device=DEV); loss = rel
        opt.zero_grad(); loss.backward(); opt.step()
        rel_s += float(rel.detach()); wall_s += float(wall.detach()); nb += 1
    sch.step()
    if (ep + 1) % 25 == 0:
        print('ep %3d  rel %.5f  wall %.5f  %5.0fs' % (ep + 1, rel_s / nb, wall_s / nb, time.time() - t0), flush=True)

torch.save({'sd': {k: v.cpu() for k, v in net.state_dict().items()}, 'sD': sD},
           os.path.join(P7, 'pub_flux_lam%.1f_s%d_m%d.pt' % (LAM, SEED, M)))

# ---------------- evaluation on the classical-cavity test split ----------------
net.eval(); outs = []
with torch.no_grad():
    for k in range(0, len(INte), 64):
        outs.append(net(INte[k:k + 64].to(DEV)).cpu() * sD)
Dp = torch.cat(outs)
Yh = Xte + DTP * Dp


def wall_q(th):
    ql = -(th[:, :, :M] * wcpu).sum(-1)
    qr = (th[:, :, -M:].flip(-1) * wcpu).sum(-1)
    return ql, qr


wcpu = torch.tensor(w.cpu().numpy())
res = {'lam': LAM, 'n_train': int(len(Xtr)), 'n_test': int(len(Xte)),
       'relL2_T': float(torch.norm(Yh[:, 2] - Yte[:, 2]) / torch.norm(Yte[:, 2])),
       'floor_relL2_T': float(torch.norm(Xte[:, 2] - Yte[:, 2]) / torch.norm(Yte[:, 2]))}
for wall, i in (('left', 0), ('right', 1)):
    qy = wall_q(Yte[:, 2])[i]
    out = {}
    for tag, Y in (('floor', Xte), ('prior', Yh)):
        qh = wall_q(Y[:, 2])[i]
        out[tag] = {'nu': float(qh.mean()),
                    'nu_err': float(((qh.mean(1) - qy.mean(1)).abs().mean() / qy.mean(1).abs().mean())),
                    'profile_err': float(((qh - qy).abs().mean() / qy.abs().mean()))}
    res[wall] = out
print(json.dumps(res, indent=1))
fn = os.path.join(OUT, 'public_flux_loss_lam%.1f_s%d_m%d.json' % (LAM, SEED, M))
json.dump(res, open(fn, 'w'), indent=1)
torch.save({'sd': {k: v.cpu() for k, v in net.state_dict().items()}, 'sD': sD},
           os.path.join(P7, 'pub_flux_lam%.1f_s%d_m%d.pt' % (LAM, SEED, M)))
print('saved', fn)

