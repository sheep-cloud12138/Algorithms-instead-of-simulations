"""First-order perturbation baseline on the 20 paired configurations.

P is the linear response of the compressible member, evaluated as a finite
difference at the smallest available contrast:
    P = (Y(0.4) - X) / 0.4 ,
which is the numerical counterpart of P = -L^{-1} S1[X] (error O(eps) in P).
Predicting Y(eps) = X + eps*P then isolates what a single linear solve buys.
"""
import json, os, re, glob
import numpy as np

DS = os.path.expanduser('~/nc_run/paired/dataset')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'perturbation.json')
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

rows, times = [], []
for k in cfg:
    d0 = tags[k][0.4]
    Y0, X = load(d0, 0.4)
    P = {c: (Y0[c] - X[c]) / 0.4 for c in CH}
    times.append(json.load(open(os.path.join(d0, 'meta.json')))['march']['comp']['wall_s'])
    for e in EPS:
        Y, Xe = load(tags[k][e], e)
        rec = dict(cfg=k, eps=e)
        for c in CH:
            pred = Xe[c] + e * P[c]
            rec[c] = float(np.linalg.norm(pred - Y[c]) / np.linalg.norm(Y[c]))
        rows.append(rec)

print('configs %d, eps %s' % (len(cfg), EPS))
for e in EPS:
    s = [r for r in rows if r['eps'] == e]
    print('eps %.1f  ' % e + '  '.join('%s %.4f' % (c, np.mean([r[c] for r in s])) for c in CH))
print('comp solve at eps=0.4: median %.1f s (n=%d)' % (np.median(times), len(times)))
json.dump(dict(n_cfg=len(cfg), eps=EPS, rows=rows,
               comp_solve_s=dict(median=float(np.median(times)), min=float(np.min(times)),
                                 max=float(np.max(times)))), open(OUT, 'w'), indent=1)
print('saved', OUT)
