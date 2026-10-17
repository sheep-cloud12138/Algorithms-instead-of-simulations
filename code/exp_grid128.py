"""Grid-independence check: same configuration at 64^2 and 128^2."""
import json, os, glob
import numpy as np

CH = ['ux', 'uy', 'theta', 'p']
EPS = [0.4, 0.8, 1.2, 1.6, 1.9]
ROOT = os.path.expanduser('~/nc_run')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'grid128.json')

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

res = {}
for res_name, root in (('64', os.path.join(ROOT, 'paired', 'dataset')),
                       ('128', os.path.join(ROOT, 'grid128', 'dataset'))):
    R = {c: [] for c in CH}
    ok = True
    for e in EPS:
        d = os.path.join(root, 'c08_Ra1e4_1100_e%g' % e, 'sample_000000')
        if not os.path.isdir(d):
            ok = False; break
        Y, X = load(d, e)
        for c in CH:
            R[c].append(float(np.linalg.norm(Y[c] - X[c]) / np.linalg.norm(Y[c])))
    if ok:
        res[res_name] = R
        print(res_name, {c: [round(v, 4) for v in R[c]] for c in CH})

if '64' in res and '128' in res:
    out = {}
    for c in CH:
        m64, b64 = np.polyfit(np.log(EPS), np.log(res['64'][c]), 1)
        m128, b128 = np.polyfit(np.log(EPS), np.log(res['128'][c]), 1)
        out[c] = dict(m64=float(m64), m128=float(m128), intercept64=float(b64), intercept128=float(b128),
                      ratio=[float(a / b) for a, b in zip(res['128'][c], res['64'][c])])
        print('%-5s m 64=%.3f 128=%.3f  max|dR|/R=%.1f%%' %
              (c, m64, m128, 100 * max(abs(1 - r) for r in out[c]['ratio'])))
    json.dump(dict(eps=EPS, R=res, fit=out), open(OUT, 'w'), indent=1)
    print('saved', OUT)
else:
    print('incomplete')
