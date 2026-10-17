"""Linear-response baseline from the paired epsilon scan.

R(eps) = (Y(eps) - X)/eps = P + eps*Q + eps^2*S + ...
P_fd   = R(0.4)                                  (O(eps) error)
P_lin  = 3R(0.4) - 3R(0.8) + R(1.2)              (O(eps^3) error)
P_lin estimates the eps -> 0 derivative, i.e. the solution of L u1 = S1[X].
Cost of that solve is measured from the Boussinesq march: wall_s / Newton iterations.
"""
import json, os, re, glob
import numpy as np

DS = os.path.expanduser('~/nc_run/paired/dataset')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'linear_response.json')
CH = ['ux', 'uy', 'theta', 'p']
EPS = [0.4, 0.8, 1.2, 1.6, 1.9]

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
cfg = [k for k in sorted(tags) if len(tags[k]) == 5]

rows = []
lin_s, comp_s = [], []
for k in cfg:
    R = {}
    for e in EPS:
        Y, X = load(tags[k][e], e)
        R[e] = {c: (Y[c] - X[c]) / e for c in CH}
    P_fd = R[0.4]
    P_lin = {c: 3 * R[0.4][c] - 3 * R[0.8][c] + R[1.2][c] for c in CH}
    for e in EPS:
        Y, X = load(tags[k][e], e)
        rec = dict(cfg=k, eps=e)
        for c in CH:
            for name, P in (('fd', P_fd), ('lin', P_lin)):
                pred = X[c] + e * P[c]
                rec['%s_%s' % (name, c)] = float(np.linalg.norm(pred - Y[c]) / np.linalg.norm(Y[c]))
        rows.append(rec)
    meta = json.load(open(os.path.join(tags[k][0.4], 'meta.json')))
    m = meta['march']
    lin_s.append(m['bouss']['wall_s'] / max(1, m['bouss']['newton_its_total']))
    comp_s.append(m['comp']['wall_s'])

print('configs %d' % len(cfg))
for e in EPS:
    s = [r for r in rows if r['eps'] == e]
    print('eps %.1f  ' % e + '  '.join(
        '%s fd %.3f lin %.3f' % (c, np.mean([r['fd_' + c] for r in s]), np.mean([r['lin_' + c] for r in s]))
        for c in CH))
print('linear solve (bouss wall_s / newton its): median %.2f s' % np.median(lin_s))
print('compressible solve: median %.1f s' % np.median(comp_s))
json.dump(dict(n_cfg=len(cfg), eps=EPS, rows=rows,
               cost=dict(linear_s=dict(median=float(np.median(lin_s)), min=float(np.min(lin_s)), max=float(np.max(lin_s))),
                         comp_s=dict(median=float(np.median(comp_s)), min=float(np.min(comp_s)), max=float(np.max(comp_s))))),
          open(OUT, 'w'), indent=1)
print('saved', OUT)
