"""POD-ROM baseline on the twelve paired 3-D samples.

Leave-one-flow-out.  For each channel, D = Y - X is represented in a POD basis
built from the eight training snapshots.  Three feature sets:
  m1 : [1]                     (mean correction)
  m2 : [1, eps]                (epsilon-aware linear ROM)
  m3 : [1, eps, X-POD coeffs]  (same information as the network)

The prior ROM is the same but the basis is built on P = D / eps and the
prediction is multiplied by eps outside; its m3 features omit eps.
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_3d_baseline import load_all, CH

LAM = 1e-4
KMAX = 7


def pod_basis(A, kmax=KMAX, energy=0.999):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    en = np.cumsum(S ** 2) / ((S ** 2).sum() + 1e-30)
    k = min(int(np.searchsorted(en, energy) + 1), kmax, len(S))
    return Vt[:k].T, U[:, :k] * S[:k], k


def fit_ridge(F, C, lam=LAM):
    mu, sd = F.mean(0), F.std(0) + 1e-12
    Fn = (F - mu) / sd
    W = np.linalg.solve(Fn.T @ Fn + lam * np.eye(Fn.shape[1]), Fn.T @ C)
    return mu, sd, W


def pred_ridge(F, mu, sd, W):
    return ((F - mu) / sd) @ W


def main():
    X, Y, eps, flow, dirs = load_all(os.path.expanduser('~/nc_run/d3/batch'))
    D = (Y - X).numpy()
    Xn = X.numpy(); Yn = Y.numpy()
    flows = np.unique(flow)

    rows = []
    for f in flows:
        tr = flow != f
        te = flow == f
        e_tr, e_te = eps[tr], eps[te]

        # X POD coefficients (same for all channels)
        Xf = Xn.transpose(0, 2, 3, 4, 1).reshape(len(eps), -1)
        Vx, _, kx = pod_basis(Xf[tr], kmax=KMAX)
        Xc = (Xf - Xf[tr].mean(0)) @ Vx

        floor = []
        for c in range(4):
            Yp = X[te, c].numpy().reshape(te.sum(), -1)
            Yt = Y[te, c].numpy().reshape(te.sum(), -1)
            floor.append(float(np.linalg.norm(Yp - Yt) / np.linalg.norm(Yt)))

        rec = {k: [] for k in ['m1', 'm2', 'm3', 'prior_m1', 'prior_m3']}
        for c in range(4):
            Dtr = D[tr, c].reshape(tr.sum(), -1)
            Dte = D[te, c].reshape(te.sum(), -1)

            # plain D basis
            Phi, C, kd = pod_basis(Dtr)
            F = {
                'm1': np.ones((tr.sum(), 1)),
                'm2': np.column_stack([np.ones(tr.sum()), e_tr]),
                'm3': np.column_stack([np.ones(tr.sum()), e_tr, Xc[tr]]),
            }
            Ft = {
                'm1': np.ones((te.sum(), 1)),
                'm2': np.column_stack([np.ones(te.sum()), e_te]),
                'm3': np.column_stack([np.ones(te.sum()), e_te, Xc[te]]),
            }
            Yt = Yn[te, c].reshape(te.sum(), -1)
            Xte = Xn[te, c].reshape(te.sum(), -1)
            for kind in ['m1', 'm2', 'm3']:
                mu, sd, W = fit_ridge(F[kind], C)
                Dp = pred_ridge(Ft[kind], mu, sd, W) @ Phi.T
                rec[kind].append(float(np.linalg.norm(Xte + Dp - Yt) / np.linalg.norm(Yt)))

            # prior P = D / eps basis
            Ptr = Dtr / e_tr[:, None]
            PhiP, CP, kp = pod_basis(Ptr)
            Fp = {
                'm1': np.ones((tr.sum(), 1)),
                'm3': np.column_stack([np.ones(tr.sum()), Xc[tr]]),
            }
            Fpt = {
                'm1': np.ones((te.sum(), 1)),
                'm3': np.column_stack([np.ones(te.sum()), Xc[te]]),
            }
            for kind, feat, featt in [('prior_m1', Fp['m1'], Fpt['m1']),
                                      ('prior_m3', Fp['m3'], Fpt['m3'])]:
                mu, sd, W = fit_ridge(feat, CP)
                Pp = pred_ridge(featt, mu, sd, W) @ PhiP.T
                Dp = e_te[:, None] * Pp
                rec[kind].append(float(np.linalg.norm(Xte + Dp - Yt) / np.linalg.norm(Yt)))

        rows.append(dict(flow=int(f), floor=floor, **rec))
        print('fold %d  floor %s' % (f, ' '.join('%.4f' % v for v in floor)))
        for kind in rec:
            print('  %-9s %s' % (kind, ' '.join('%s=%.4f' % (CH[i], rec[kind][i]) for i in range(4))))

    agg = {}
    for kind in ['floor'] + [k for k in rows[0] if k not in ('flow', 'floor')]:
        A = np.array([r[kind] if kind != 'floor' else r['floor'] for r in rows])
        agg[kind] = dict(mean=A.mean(0).tolist(), std=A.std(0).tolist())
    plain = np.array(agg['m3']['mean'])
    for kind in agg:
        if kind not in ('floor', 'm3'):
            agg[kind]['gain_vs_m3'] = (plain / np.array(agg[kind]['mean'])).tolist()

    print('\n=== mean over 3 folds ===')
    print('%-10s %s' % ('variant', ' '.join('%9s' % c for c in CH)))
    for kind in ['floor', 'm1', 'm2', 'm3', 'prior_m1', 'prior_m3']:
        m, s = agg[kind]['mean'], agg[kind]['std']
        print('%-10s %s' % (kind, ' '.join('%5.4f±%.3f' % (m[i], s[i]) for i in range(4))))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       '论文', '3d_rom.json')
    json.dump(dict(samples=dirs, eps=eps.tolist(), flow=flow.tolist(),
                   rows=rows, aggregate=agg), open(out, 'w'), indent=1)
    print('\nsaved', out)


if __name__ == '__main__':
    main()
