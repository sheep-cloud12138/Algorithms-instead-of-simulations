"""3-D surrogate baseline on the twelve paired 3-D samples.

Leave-one-flow-out cross-validation (3 flows x 4 eps).  Variants:
  plain   : UNet3 predicts D = Y - X directly
  eps     : same, with eps as a constant input channel
  prior   : D = eps * P(X), eps outside the network
  prior2  : D = (eps + beta*eps^2) * P(X)

Usage:
  python exp_3d_baseline.py --smoke
  python exp_3d_baseline.py --epochs 200 --seeds 0,1,2
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CH = ['ux', 'uy', 'theta', 'p']
TAGS = ['e02', 'e04', 'e08', 'e12']


def to_grid(coords, vals, n):
    idx, axes = [], []
    for d in range(3):
        u = np.unique(np.round(coords[:, d], 10))
        if len(u) != n:
            raise ValueError('axis %d has %d points, expected %d' % (d, len(u), n))
        idx.append(np.searchsorted(u, np.round(coords[:, d], 10)))
        axes.append(u)
    arr = np.empty((n, n, n), dtype=np.float64)
    arr[idx[0], idx[1], idx[2]] = vals
    if len(np.unique(idx[0] * n * n + idx[1] * n + idx[2])) != n ** 3:
        raise ValueError('grid mapping is not one-to-one')
    return arr


def deramp_y(vals, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return vals - A @ c


def load_field(d, name, n, coord_name=None, even=False):
    coord_name = coord_name or name
    c = np.load(os.path.join(d, 'coords_' + coord_name + '.npy'))
    v = np.load(os.path.join(d, name + '.npy')).astype(np.float64)
    a = to_grid(c, v, n)
    return a[::2, ::2, ::2] if even else a


def load_pair(d, eps):
    Tb = load_field(d, 'T_bouss', 17)
    Tc = load_field(d, 'T_comp', 17)
    thb = (Tb - (1.0 - eps / 2.0)) / eps
    thc = (Tc - (1.0 - eps / 2.0)) / eps

    ub = np.empty((4, 17, 17, 17), np.float64)
    uc = np.empty((4, 17, 17, 17), np.float64)
    ub[0] = load_field(d, 'ux_bouss', 33, even=True)
    ub[1] = load_field(d, 'uy_bouss', 33, even=True)
    ub[2] = thb
    uc[0] = load_field(d, 'ux_comp', 33, even=True)
    uc[1] = load_field(d, 'uy_comp', 33, even=True)
    uc[2] = thc

    cb = np.load(os.path.join(d, 'coords_p_bouss.npy'))
    cc = np.load(os.path.join(d, 'coords_p_comp.npy'))
    pb = -deramp_y(np.load(os.path.join(d, 'p_bouss_pert_raw.npy')), cb[:, 1])
    pc = deramp_y(np.load(os.path.join(d, 'p_comp_pert_raw.npy')), cc[:, 1])
    ub[3] = to_grid(cb, pb, 17)
    uc[3] = to_grid(cc, pc, 17)
    return ub.astype(np.float32), uc.astype(np.float32)


def load_all(root):
    Xs, Ys, eps, flow, dirs = [], [], [], [], []
    for tag in TAGS:
        for d in sorted(glob.glob(os.path.join(root, tag, 'dataset', 'sample_*'))):
            need = ['ux_comp.npy', 'uy_comp.npy', 'T_comp.npy', 'meta.json']
            if not all(os.path.exists(os.path.join(d, x)) for x in need):
                continue
            m = json.load(open(os.path.join(d, 'meta.json')))
            e = float(m['dT'])
            X, Y = load_pair(d, e)
            Xs.append(X); Ys.append(Y)
            eps.append(e); flow.append(int(m['seed'])); dirs.append(d)
    if not Xs:
        raise RuntimeError('no complete 3-D samples under ' + root)
    order = np.lexsort((np.array(eps), np.array(flow)))
    X = torch.tensor(np.stack(Xs)[order], dtype=torch.float32)
    Y = torch.tensor(np.stack(Ys)[order], dtype=torch.float32)
    eps = np.array(eps, np.float32)[order]
    flow = np.array(flow, np.int64)[order]
    dirs = [dirs[i] for i in order]
    return X, Y, eps, flow, dirs


class Blk3(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c1 = nn.Conv3d(ci, co, 3, padding=1)
        self.n1 = nn.GroupNorm(4, co)
        self.c2 = nn.Conv3d(co, co, 3, padding=1)
        self.n2 = nn.GroupNorm(4, co)

    def forward(self, x):
        return F.silu(self.n2(self.c2(F.silu(self.n1(self.c1(x))))))


class UNet3(nn.Module):
    def __init__(self, ci, w=16):
        super().__init__()
        self.e1 = Blk3(ci, w)
        self.e2 = Blk3(w, 2 * w)
        self.e3 = Blk3(2 * w, 4 * w)
        self.b = Blk3(4 * w, 4 * w)
        self.d3 = Blk3(4 * w + 4 * w, 2 * w)
        self.d2 = Blk3(2 * w + 2 * w, w)
        self.d1 = Blk3(w + w, w)
        self.o = nn.Conv3d(w, 4, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(F.max_pool3d(x1, 2))
        x3 = self.e3(F.max_pool3d(x2, 2))
        b = self.b(F.max_pool3d(x3, 2))
        y = F.interpolate(b, size=x3.shape[-3:], mode='trilinear', align_corners=False)
        y = self.d3(torch.cat([y, x3], 1))
        y = F.interpolate(y, size=x2.shape[-3:], mode='trilinear', align_corners=False)
        y = self.d2(torch.cat([y, x2], 1))
        y = F.interpolate(y, size=x1.shape[-3:], mode='trilinear', align_corners=False)
        y = self.d1(torch.cat([y, x1], 1))
        return self.o(y)


class Plain3(nn.Module):
    def __init__(self, eps_in, w=16):
        super().__init__()
        self.eps_in = eps_in
        self.base = UNet3(5 if eps_in else 4, w)

    def forward(self, x, e):
        if self.eps_in:
            e = e.expand(-1, -1, x.shape[-3], x.shape[-2], x.shape[-1])
            return self.base(torch.cat([x, e], 1))
        return self.base(x)


class Prior3(nn.Module):
    def __init__(self, quad, w=16):
        super().__init__()
        self.quad = quad
        self.base = UNet3(4, w)
        self.beta = nn.Parameter(torch.zeros(4))

    def forward(self, x, e):
        p = self.base(x)
        if self.quad:
            s = e + self.beta.view(1, 4, 1, 1, 1) * e ** 2
        else:
            s = e
        return s * p


def make_net(kind, w):
    if kind == 'plain':
        return Plain3(False, w)
    if kind == 'eps':
        return Plain3(True, w)
    if kind == 'prior':
        return Prior3(False, w)
    if kind == 'prior2':
        return Prior3(True, w)
    raise ValueError(kind)


def rel_l2(Yp, Yt):
    return [float((((Yp[:, c] - Yt[:, c]) ** 2).sum() /
                   ((Yt[:, c] ** 2).sum() + 1e-12)) ** 0.5) for c in range(4)]


def run_one(kind, X, Y, eps, tr, te, seed, epochs, w, dev='cuda', bs=2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sx = X[tr].reshape(int(tr.sum()), 4, -1).std(2).mean(0).view(1, 4, 1, 1, 1)
    Xn = X / sx
    D = Y - X
    sd = D[tr].reshape(int(tr.sum()), 4, -1).std(2).mean(0).view(1, 4, 1, 1, 1)
    Dn = D / sd
    e = torch.tensor(eps).view(-1, 1, 1, 1, 1)

    net = make_net(kind, w).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx = np.where(tr)[0].copy()
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        np.random.shuffle(idx)
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            pr = net(Xn[b].to(dev), e[b].to(dev))
            dt = Dn[b].to(dev)
            rel = ((pr - dt).pow(2).mean((1, 2, 3, 4)) /
                   (dt.pow(2).mean((1, 2, 3, 4)) + 1e-12)).mean()
            opt.zero_grad()
            rel.backward()
            opt.step()
        sch.step()

    net.eval()
    out = []
    ii = np.where(te)[0]
    with torch.no_grad():
        for k in range(0, len(ii), 4):
            out.append(net(Xn[ii[k:k + 4]].to(dev), e[ii[k:k + 4]].to(dev)).cpu() * sd)
    Yp = X[te] + torch.cat(out)
    return rel_l2(Yp, Y[te]), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/nc_run/d3/batch'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--width', type=int, default=16)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--variants', default='plain,eps,prior,prior2')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    X, Y, eps, flow, dirs = load_all(args.root)
    flows = np.unique(flow)
    print('%d samples, %d flows, eps=%s' % (len(eps), len(flows), sorted(set(eps))))
    print('flows', flows.tolist())
    print('channels', CH)

    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    variants = [x.strip() for x in args.variants.split(',') if x.strip()]
    if args.smoke:
        flows = flows[:1]
        seeds = seeds[:1]
        variants = variants[:2]
        args.epochs = min(args.epochs, 2)
        print('SMOKE: one fold, one seed, %d epochs' % args.epochs)

    rows = []
    t_all = time.time()
    for f in flows:
        tr = flow != f
        te = flow == f
        floor = rel_l2(X[te], Y[te])
        print('\nfold flow=%d train=%d test=%d floor=%s' %
              (f, tr.sum(), te.sum(), ' '.join('%.4f' % v for v in floor)))
        rows.append(dict(kind='floor', flow=int(f), seed=-1, g=floor, sec=0.0))
        for seed in seeds:
            for kind in variants:
                g, sec = run_one(kind, X, Y, eps, tr, te, seed,
                                 args.epochs, args.width)
                rows.append(dict(kind=kind, flow=int(f), seed=seed, g=g, sec=sec))
                print('  %-7s seed=%d  %s  %5.1fs' %
                      (kind, seed, ' '.join('%s=%.4f' % (CH[i], g[i]) for i in range(4)), sec),
                      flush=True)

    # aggregate
    agg = {}
    for kind in ['floor'] + variants:
        rr = [r for r in rows if r['kind'] == kind]
        G = np.array([r['g'] for r in rr], float)
        agg[kind] = dict(mean=G.mean(0).tolist(), std=G.std(0).tolist(), n=len(rr))
    plain = np.array(agg['plain']['mean'])
    for kind in variants:
        if kind != 'plain':
            agg[kind]['gain_vs_plain'] = (plain / np.array(agg[kind]['mean'])).tolist()

    print('\n=== mean over folds/seeds ===')
    print('%-8s %s' % ('variant', ' '.join('%9s' % c for c in CH)))
    for kind in ['floor'] + variants:
        m = agg[kind]['mean']; s = agg[kind]['std']
        print('%-8s %s' % (kind, ' '.join('%5.4f±%.3f' % (m[i], s[i]) for i in range(4))))

    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   '论文', '3d_baseline.json')
    payload = dict(
        root=args.root, epochs=args.epochs, width=args.width,
        seeds=seeds, variants=variants, samples=dirs, eps=eps.tolist(),
        flow=flow.tolist(), aggregate=agg, runs=rows,
        wall_sec=time.time() - t_all,
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, 'w'), indent=1)
    print('\nsaved', out, 'wall %.1fs' % payload['wall_sec'])


if __name__ == '__main__':
    main()
