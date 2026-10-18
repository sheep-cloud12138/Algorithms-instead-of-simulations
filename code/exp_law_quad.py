"""Quadratic vs power-law description of R(eps) on the 627 points."""
import json, os
import numpy as np
from scipy import stats

z = np.load(os.path.expanduser('~/nc_run/p7/eps_law_R.npz'))
e = z['eps_all']; R = z['R_all']
CH = ['ux', 'uy', 'theta', 'p']
out = {}
for k, c in enumerate(CH):
    y = R[:, k]
    m, b, rv, pv, se = stats.linregress(np.log(e), np.log(y))
    A2 = np.column_stack([e, e ** 2])
    coef, *_ = np.linalg.lstsq(A2, y, rcond=None)
    fit2 = A2 @ coef
    r2_2 = 1 - ((y - fit2) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    fit1 = coef[0] * e
    r2_1 = 1 - ((y - fit1) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rel2 = np.abs(fit2 - y) / y
    out[c] = dict(power=dict(m=float(m), se=float(se), r2=float(rv ** 2)),
                  quad=dict(a=float(coef[0]), c=float(coef[1]), kappa=float(coef[1] / coef[0]),
                            r2=float(r2_2), rel_med=float(np.median(rel2)),
                            rel_p90=float(np.percentile(rel2, 90))),
                  linear=dict(r2=float(r2_1)))
    print('%-5s power m=%.3f (R2 %.3f) | quad a=%.4f c=%.4f kappa=%.3f (R2 %.3f, med rel %.3f, p90 %.3f) | pure-linear R2 %.3f'
          % (c, m, rv ** 2, coef[0], coef[1], coef[1] / coef[0], r2_2,
             np.median(rel2), np.percentile(rel2, 90), r2_1))

print('\nper-decade check of the quadratic form, ux:')
c = 'ux'; y = R[:, 0]
a, cc = out[c]['quad']['a'], out[c]['quad']['c']
for lo, hi in ((0.02, 0.1), (0.1, 0.4), (0.4, 1.2), (1.2, 1.9)):
    s = (e >= lo) & (e <= hi)
    print('  eps %.2f-%.2f  n=%3d  observed/predicted mean %.3f' %
          (lo, hi, s.sum(), (y[s] / (a * e[s] + cc * e[s] ** 2)).mean()))

json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'law_quad.json'), 'w'), indent=1)