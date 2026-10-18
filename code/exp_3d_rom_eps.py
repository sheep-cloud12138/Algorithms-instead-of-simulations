"""POD-ROM baseline on 3-D samples, leave-one-epsilon-out.

Train on 3 flows x 3 eps (9 samples), test on the held-out eps (3 samples).
This tests the quantity the paper actually claims: transfer across temperature
contrast, not across a new boundary-condition family.
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_3d_baseline import load_all, CH
from exp_3d_rom import pod_basis, fit_ridge, pred_ridge

LAM = 1e-4
KMAX = 4


def main():
    X, Y, eps, flow, dirs = load_all(os.path.expanduser('~/nc_run/d3/batch'))
    D = (Y - X).numpy()
    Xn = X.numpy(); Yn = Y.numpy()
    rows = []
    for e0 in np.unique(eps):
        tr = eps != e0
        te = eps == e0
        e_tr, e_te = eps[tr], eps[te]

        Xf = Xn.transpose(0, 2, 3, 4, 1).reshape(len(eps), -1)
        Vx, _, kx = pod_basis(Xf[tr], kmax=KMAX, energy=1.0)
        Xc = (Xf - Xf[tr].mean(0)) @ Vx

        rec = {k: [] for k in ['m1', 'm2', 'm3', 'prior_m1', 'prior_m2', 'prior_m3']}
        floor = []
        for c in range(4):
            Dtr = D[tr, c].reshape(tr.sum(), -1)
            Xte = Xn[te, c].reshape(te.sum(), -1)
            Yt = Yn[te, c].reshape(te.sum(), -1)
            floor.append(float(np.linalg.norm(Xte - Yt) / np.linalg.norm(Yt)))

            Phi, C, _ = pod_basis(Dtr, energy=1.0)
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
            for kind in ['m1', 'm2', 'm3']:
                mu, sd, W = fit_ridge(F[kind], C)
                Dp = pred_ridge(Ft[kind], mu, sd, W) @ Phi.T
                rec[kind].append(float(np.linalg.norm(Xte + Dp - Yt) / np.linalg.norm(Yt)))

            Ptr = Dtr / e_tr[:, None]
            PhiP, CP, _ = pod_basis(Ptr, energy=1.0)
            Fp = {
                'm1': np.ones((tr.sum(), 1)),
                'm2': np.column_stack([np.ones(tr.sum()), e_tr]),
                'm3': np.column_stack([np.ones(tr.sum()), Xc[tr]]),
            }
            Fpt = {
                'm1': np.ones((te.sum(), 1)),
                'm2': np.column_stack([np.ones(te.sum()), e_te]),
                'm3': np.column_stack([np.ones(te.sum()), Xc[te]]),
            }
            for kind, feat, featt in [('prior_m1', Fp['m1'], Fpt['m1']),
                                      ('prior_m2', Fp['m2'], Fpt['m2']),
                                      ('prior_m3', Fp['m3'], Fpt['m3'])]:
                mu, sd, W = fit_ridge(feat, CP)
                Pp = pred_ridge(featt, mu, sd, W) @ PhiP.T
                Dp = e_te[:, None] * Pp
                rec[kind].append(float(np.linalg.norm(Xte + Dp - Yt) / np.linalg.norm(Yt)))

        rows.append(dict(eps=float(e0), n_train=int(tr.sum()), n_test=int(te.sum()),
                         floor=floor, **rec))
        print('hold eps %.1f  floor %s' % (e0, ' '.join('%.4f' % v for v in floor)))
        for kind in rec:
            print('  %-9s %s' % (kind, ' '.join('%s=%.4f' % (CH[i], rec[kind][i]) for i in range(4))))

    agg = {}
    for kind in ['floor', 'm1', 'm2', 'm3', 'prior_m1', 'prior_m2', 'prior_m3']:
        A = np.array([r[kind] for r in rows])
        agg[kind] = dict(mean=A.mean(0).tolist(), std=A.std(0).tolist())
    ref = np.array(agg['m1']['mean'])
    for kind in agg:
        if kind != 'm1':
            agg[kind]['gain_vs_m1'] = (ref / np.array(agg[kind]['mean'])).tolist()
    ref3 = np.array(agg['m3']['mean'])
    for kind in ['prior_m1', 'prior_m2', 'prior_m3', 'floor', 'm1', 'm2']:
        agg[kind]['gain_vs_m3'] = (ref3 / np.array(agg[kind]['mean'])).tolist()

    print('\n=== mean over %d held-out eps ===' % len(rows))
    print('%-10s %s' % ('variant', ' '.join('%9s' % c for c in CH)))
    for kind in ['floor', 'm1', 'm2', 'm3', 'prior_m1', 'prior_m2', 'prior_m3']:
        m, s = agg[kind]['mean'], agg[kind]['std']
        print('%-10s %s' % (kind, ' '.join('%5.4f±%.3f' % (m[i], s[i]) for i in range(4))))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       '论文', '3d_rom_eps.json')
    json.dump(dict(samples=dirs, eps=eps.tolist(), flow=flow.tolist(),
                   rows=rows, aggregate=agg), open(out, 'w'), indent=1)
    print('\nsaved', out)


if __name__ == '__main__':
    main()
