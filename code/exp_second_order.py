"""Second-order perturbation baseline from the paired epsilon scan.

R(eps) = (Y(eps) - X)/eps = P + eps*Q + O(eps^2).
P and Q are fitted by least squares on eps = 0.4, 0.8, 1.2 and tested at
1.6 and 1.9.  This is a data-driven two-term expansion, not a linear solve;
it answers whether the measured O(eps^2) term closes the gap left by P alone.
"""
import glob
import json
import os
import re

import numpy as np

DS = os.path.expanduser('~/nc_run/paired/dataset')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   '论文', 'second_order.json')
CH = ['ux', 'uy', 'theta', 'p']
EPS = [0.4, 0.8, 1.2, 1.6, 1.9]
FIT = [0.4, 0.8, 1.2]


def deramp(v, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ c


def load(d, eps):
    g = lambda n: np.load(os.path.join(d, n + '.npy')).astype(np.float64)
    Tw = lambda T: (T - (1.0 - eps / 2.0)) / eps
    y = g('coords_p_bouss')[:, 1]
    return {'ux': g('ux_comp'), 'uy': g('uy_comp'), 'theta': Tw(g('T_comp')),
            'p': deramp(g('p_comp_pert_raw'), y)}, \
           {'ux': g('ux_bouss'), 'uy': g('uy_bouss'), 'theta': Tw(g('T_bouss')),
            'p': -deramp(g('p_bouss_pert_raw'), y)}


tags = {}
for d in sorted(glob.glob(os.path.join(DS, 'c*_Ra*'))):
    m = re.match(r'(c\d+_Ra\S+?_\d{4})_e([\d.]+)$', os.path.basename(d))
    if m and os.path.isdir(os.path.join(d, 'sample_000000')):
        tags.setdefault(m.group(1), {})[float(m.group(2))] = os.path.join(d, 'sample_000000')
cfg = [k for k in sorted(tags) if all(e in tags[k] for e in EPS)]

rows = []
for k in cfg:
    R = {}
    for e in EPS:
        Y, X = load(tags[k][e], e)
        R[e] = {c: (Y[c] - X[c]) / e for c in CH}

    # Least-squares fit R(eps) = P + eps*Q on the three lowest contrasts.
    A = np.column_stack([np.ones(len(FIT)), FIT])
    coef = {c: np.linalg.lstsq(A, np.array([R[e][c] for e in FIT]), rcond=None)[0]
            for c in CH}
    P = {c: coef[c][0] for c in CH}
    Q = {c: coef[c][1] for c in CH}
    P_fd = {c: R[0.4][c] for c in CH}
    P_lin = {c: 3 * R[0.4][c] - 3 * R[0.8][c] + R[1.2][c] for c in CH}

    for e in EPS:
        Y, X = load(tags[k][e], e)
        rec = dict(cfg=k, eps=e)
        for c in CH:
            for name, pred in (
                    ('fd', X[c] + e * P_fd[c]),
                    ('lin', X[c] + e * P_lin[c]),
                    ('p2', X[c] + e * P[c] + e * e * Q[c])):
                rec[name + '_' + c] = float(np.linalg.norm(pred - Y[c]) / np.linalg.norm(Y[c]))
        rows.append(rec)

print('configs %d, fit %s, test %s' % (len(cfg), FIT, EPS))
for e in EPS:
    s = [r for r in rows if r['eps'] == e]
    print('eps %.1f  ' % e + '  '.join(
        '%s fd %.3f lin %.3f p2 %.3f' % (
            c,
            np.mean([r['fd_' + c] for r in s]),
            np.mean([r['lin_' + c] for r in s]),
            np.mean([r['p2_' + c] for r in s]))
        for c in CH))

json.dump(dict(n_cfg=len(cfg), fit=FIT, eps=EPS, rows=rows), open(OUT, 'w'), indent=1)
print('saved', OUT)
