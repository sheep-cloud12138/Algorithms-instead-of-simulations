"""Grid-independence check: the same configuration at 64^2 and 128^2."""
import json, os, glob
import numpy as np

CH = ['ux', 'uy', 'theta', 'p']
ROOT = os.path.expanduser('~/nc_run')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'grid128.json')

CFG = {
    'Ra1e4': dict(
        eps=[0.4, 0.8, 1.2, 1.6, 1.9],
        d64=lambda e: os.path.join(ROOT, 'paired', 'dataset', 'c08_Ra1e4_1100_e%g' % e),
        d128=lambda e: os.path.join(ROOT, 'grid128', 'dataset', 'c08_Ra1e4_1100_e%g' % e)),
    'Ra1e6': dict(
        eps=[0.4, 1.2, 1.9],
        d64=lambda e: os.path.join(ROOT, 'paired', 'dataset', 'c16_Ra1e6_1100_e%g' % e),
        d128=lambda e: os.path.join(ROOT, 'grid_ra1e6', 'd128' if e == 1.9 else 'd128_e%g' % e)),
}

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

out = {}
for name, cfg in CFG.items():
    EPS = cfg['eps']
    res = {}
    for tag, key in (('64', 'd64'), ('128', 'd128')):
        R = {c: [] for c in CH}
        for e in EPS:
            d = os.path.join(cfg[key](e), 'sample_000000')
            Y, X = load(d, e)
            for c in CH:
                R[c].append(float(np.linalg.norm(Y[c] - X[c]) / np.linalg.norm(Y[c])))
        res[tag] = R
        print(name, tag, {c: [round(v, 4) for v in R[c]] for c in CH})
    fit = {}
    for c in CH:
        m64, b64 = np.polyfit(np.log(EPS), np.log(res['64'][c]), 1)
        m128, b128 = np.polyfit(np.log(EPS), np.log(res['128'][c]), 1)
        fit[c] = dict(m64=float(m64), m128=float(m128), intercept64=float(b64), intercept128=float(b128),
                      ratio=[float(a / b) for a, b in zip(res['128'][c], res['64'][c])])
        print('%-5s m 64=%.3f 128=%.3f  max|dR|/R=%.1f%%' %
              (c, m64, m128, 100 * max(abs(1 - r) for r in fit[c]['ratio'])))
    out[name] = dict(eps=EPS, R=res, fit=fit)

json.dump(out, open(OUT, 'w'), indent=1)
print('saved', OUT)
