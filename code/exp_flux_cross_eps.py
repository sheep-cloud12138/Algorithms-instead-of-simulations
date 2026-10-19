"""Cross-eps wall-Nu test of the public single-eps models.

Models are trained on the public 4000 random-wall pairs at eps=1.2.
Test set: our 600 paired samples, binned by eps.  Only samples with
|Nu_Y|>1 at the left wall are used, because the relative Nu error is
undefined when the denominator is small (same filter as exp_dnu_clean).
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600
sys.path.insert(0, os.path.expanduser('~/nc_run/p4'))
from common import UNet

PUB = os.path.expanduser('~/nc_run/pub')
P7 = os.path.expanduser('~/nc_run/p7')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'flux_cross_eps.json')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
M = 6
BINS = [(0.35, 0.45, '0.4'), (0.70, 0.90, '0.8'), (1.50, 1.70, '1.6'), (1.85, 1.95, '1.9')]


def theta(T, eps):
    return (T - (1.0 - eps / 2.0)) / eps


def stencil(n=64):
    xw = np.arange(M) * (1.0 / (n - 1))
    return np.linalg.solve(np.vander(xw, M, increasing=True).T, np.eye(M)[1])


W = stencil()


def wall_nu(th):
    ql = -np.einsum('nyi,i->ny', th[:, :, :M], W)
    qr = np.einsum('nyi,i->ny', th[:, :, -M:][:, :, ::-1], W)
    return ql.mean(1), qr.mean(1)


# public normalisation, identical to exp_public_nu.py
Xp = torch.load(PUB + '/train/X.pt').float()
Xp[:, 2] = theta(Xp[:, 2], 1.2)
sx = Xp.reshape(len(Xp), 4, -1).std(2).mean(0).view(1, 4, 1, 1)

X, Y, eps, meta = load600()
E = torch.tensor(eps, dtype=torch.float32).view(-1, 1, 1, 1)
IN = (X / sx).to(DEV)
print('device', DEV, 'n', len(X))

cache = torch.load(P7 + '/pub_nets.pt', weights_only=False)
nets = {}
for key in ('pub_naive', 'pub_prior'):
    net = UNet(4, 32).to(DEV); net.load_state_dict(cache[key]['sd']); net.eval()
    nets[key] = (net, cache[key]['sD'])

flux_paths = [P7 + '/pub_flux_lam1.0.pt', P7 + '/pub_flux_lam1.0_s1.pt']
flux = []
for p in flux_paths:
    c = torch.load(p, weights_only=False)
    net = UNet(4, 32).to(DEV); net.load_state_dict(c['sd']); net.eval()
    flux.append((net, float(c['sD'])))

pred = {}
with torch.no_grad():
    outs = {}
    for key in ('pub_naive', 'pub_prior'):
        net, sD = nets[key]
        outs[key] = torch.cat([net(IN[k:k + 64]).cpu() for k in range(0, len(IN), 64)])
    pred['plain'] = X + outs['pub_naive'] * nets['pub_naive'][1]
    pred['prior'] = X + E * outs['pub_prior'] * nets['pub_prior'][1]
    fout = []
    for net, sD in flux:
        fout.append(torch.cat([net(IN[k:k + 64]).cpu() for k in range(0, len(IN), 64)]) * sD)
    pred['prior+flux'] = X + E * torch.stack(fout).mean(0)

Ynp = Y.numpy()
res = {'n_total': int(len(X)), 'bins': {}}
rng = np.random.default_rng(0)

def boot(a, b, n=4000):
    d = []
    for _ in range(n):
        k = rng.integers(0, len(a), len(a))
        d.append(a[k].mean() - b[k].mean())
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

for lo, hi, tag in BINS:
    sel = (eps >= lo) & (eps <= hi)
    if sel.sum() == 0:
        continue
    row = {'n_bin': int(sel.sum())}
    nuY_l, nuY_r = wall_nu(Ynp[sel, 2])
    keep = np.abs(nuY_l) > 1.0
    row['n_kept'] = int(keep.sum())
    if keep.sum() == 0:
        res['bins'][tag] = row
        continue
    row['nuY_mean'] = float(nuY_l[keep].mean())
    errs = {}
    for name, arr in [('do-nothing', X.numpy())] + [(k, v.numpy()) for k, v in pred.items()]:
        nl, nr = wall_nu(arr[sel, 2])
        e_l = np.abs(nl - nuY_l) / np.abs(nuY_l)
        e_r = np.abs(nr - nuY_r) / np.abs(nuY_r)
        errs[name] = e_l
        row[name] = {'left': float(e_l[keep].mean()),
                     'left_sem': float(e_l[keep].std(ddof=1) / np.sqrt(keep.sum())),
                     'both': float(0.5 * (e_l[keep] + e_r[keep]).mean()),
                     'median_left': float(np.median(e_l[keep]))}
    row['flux_vs_prior'] = boot(errs['prior+flux'][keep], errs['prior'][keep])
    row['flux_vs_floor'] = boot(errs['prior+flux'][keep], errs['do-nothing'][keep])
    res['bins'][tag] = row
    print(tag, 'n=%d kept=%d' % (sel.sum(), keep.sum()),
          ' | '.join('%s %.3f%%' % (k, 100 * row[k]['left'])
                     for k in ('do-nothing', 'plain', 'prior', 'prior+flux')),
          '| flux-prior', '[%.2f, %.2f] pp' % tuple(100 * v for v in row['flux_vs_prior']),
          flush=True)

json.dump(res, open(OUT, 'w'), indent=1)
print('saved', OUT)
