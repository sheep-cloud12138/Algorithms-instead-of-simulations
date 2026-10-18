"""Wall and interior heat transfer of the public-data surrogate: 4000 train -> 200 test, eps=1.2."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import build_input
sys.path.insert(0, os.path.expanduser('~/nc_run/p4'))
from common import UNet

PUB = os.path.expanduser('~/nc_run/pub')
P7 = os.path.expanduser('~/nc_run/p7')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'public_nu.json')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DTP = 1.2
PR = 0.71
M = 6


def theta(T):
    return (T - (1.0 - DTP / 2.0)) / DTP


def stencil(n):
    xw = np.arange(M) * (1.0 / (n - 1))
    return np.linalg.solve(np.vander(xw, M, increasing=True).T, np.eye(M)[1])


def wall_flux(th):
    w = stencil(th.shape[-1])
    ql = -np.einsum('nyi,i->ny', th[:, :, :M], w)
    qr = np.einsum('nyi,i->ny', th[:, :, -M:][:, :, ::-1], w)
    return ql, qr


def interior_flux(A):
    """Vertically integrated heat flux at x=0.25, 0.5, 0.75; mean over the three planes."""
    ux, t = A[:, 0], A[:, 2]
    gx = np.gradient(t, axis=2) * (t.shape[2] - 1)
    vals = []
    for j in (16, 32, 48):
        vals.append((np.sqrt(RA * PR) * ux[:, :, j] * t[:, :, j] - gx[:, :, j]).mean(1))
    return np.mean(vals, 0)


Xtr = torch.load(PUB + '/train/X.pt').float()
Ytr = torch.load(PUB + '/train/Y.pt').float()
Xte = torch.load(PUB + '/test_legacy/X.pt').float()
Yte = torch.load(PUB + '/test_legacy/Y.pt').float()
import csv
rows = list(csv.DictReader(open(PUB + '/test_legacy/per_sample.csv')))
RA = np.array([[float(r['Ra'])] for r in rows])
Xtr[:, 2], Ytr[:, 2] = theta(Xtr[:, 2]), theta(Ytr[:, 2])
Xte[:, 2], Yte[:, 2] = theta(Xte[:, 2]), theta(Yte[:, 2])
INtr, sx = build_input(Xtr, np.full(len(Xtr), DTP, np.float32), False)
INte = Xte / sx

cache = torch.load(P7 + '/pub_nets.pt', weights_only=False)
preds = {}
for key in ('pub_naive', 'pub_prior'):
    net = UNet(4, 32).to(DEV)
    net.load_state_dict(cache[key]['sd'])
    net.eval()
    sD = cache[key]['sD']
    out = []
    with torch.no_grad():
        for k in range(0, len(INte), 32):
            out.append(net(INte[k:k + 32].to(DEV)).cpu() * sD)
    Dp = torch.cat(out)
    preds[key] = (Xte + Dp) if key == 'pub_naive' else (Xte + DTP * Dp)

res = dict(n_train=int(len(Xtr)), n_test=int(len(Xte)), eps=DTP,
           test_relL2_T=float(torch.norm(preds['pub_prior'][:, 2] - Yte[:, 2]) / torch.norm(Yte[:, 2])),
           floor_relL2_T=float(torch.norm(Xte[:, 2] - Yte[:, 2]) / torch.norm(Yte[:, 2])))
for wall, i in (('left', 0), ('right', 1)):
    qx = wall_flux(Xte[:, 2].numpy())[i]
    qy = wall_flux(Yte[:, 2].numpy())[i]
    r = {}
    for key in ('pub_naive', 'pub_prior'):
        qh = wall_flux(preds[key][:, 2].numpy())[i]
        e = np.abs(qh - qy)
        r[key] = dict(nu_mean=float(qh.mean()),
                      nu_err=float(np.abs(qh.mean(1) - qy.mean(1)).mean() / np.abs(qy.mean(1)).mean()),
                      profile_err=float(e.mean() / np.abs(qy).mean()),
                      profile_p90=float(np.percentile(e, 90) / np.abs(qy).mean()))
    e = np.abs(qx - qy)
    r['floor'] = dict(nu_mean=float(qx.mean()),
                      nu_err=float(np.abs(qx.mean(1) - qy.mean(1)).mean() / np.abs(qy.mean(1)).mean()),
                      profile_err=float(e.mean() / np.abs(qy).mean()),
                      profile_p90=float(np.percentile(e, 90) / np.abs(qy).mean()))
    res[wall] = r

Fy = interior_flux(Yte.numpy())
Fx = interior_flux(Xte.numpy())
res['interior_flux'] = dict(nu_y_mean=float(Fy.mean()))
for key in ('pub_naive', 'pub_prior'):
    Fh = interior_flux(preds[key].numpy())
    res['interior_flux'][key] = dict(mean=float(Fh.mean()),
                                      err=float(np.mean(np.abs(Fh - Fy)) / np.abs(Fy).mean()))
res['interior_flux']['floor'] = dict(mean=float(Fx.mean()),
                                     err=float(np.mean(np.abs(Fx - Fy)) / np.abs(Fy).mean()))

json.dump(res, open(OUT, 'w'), indent=1)
print('public test n=%d, eps=%.1f' % (len(Xte), DTP))
print('relL2 T: floor %.4f  prior %.4f' % (res['floor_relL2_T'], res['test_relL2_T']))
for wall in ('left', 'right'):
    print('\nwall', wall)
    for tag in ('floor', 'pub_naive', 'pub_prior'):
        r = res[wall][tag]
        print('  %-10s Nu %.4f  |dNu|/Nu %.4f  profile err %.4f  p90 %.4f'
              % (tag, r['nu_mean'], r['nu_err'], r['profile_err'], r['profile_p90']))
print('\ninterior flux: Y %.4f' % res['interior_flux']['nu_y_mean'])
for tag in ('floor', 'pub_naive', 'pub_prior'):
    r = res['interior_flux'][tag]
    print('  %-10s mean %.4f  err %.4f' % (tag, r['mean'], r['err']))
print('\nsaved', OUT)
