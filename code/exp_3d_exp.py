"""3-D exponents at matched eps windows and matched seeds.

Usage: python3 exp_3d_exp.py [--eps 0.4,1.2,1.9] [--json out.json] root [root ...]

Each root holds tag dirs (c<N>_e<eps>/sample_000000 or e<eps>/dataset/sample_*).
Samples are grouped by (Ra, seed) so one flow keeps the same wall pattern.  Two fits
are reported: a separate power law per flow, and the model the paper quotes,
R_f = A_f eps^m, with one prefactor per flow and one exponent per Rayleigh number.
"""
import argparse, glob, json, os
import numpy as np

CH = ('ux', 'uy', 'theta', 'p')


def deramp_y(vals, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return vals - A @ c


def load(d, eps):
    g = lambda n: np.load(os.path.join(d, n + '.npy')).astype(np.float64)
    Tb, Tc = g('T_bouss'), g('T_comp')
    y = g('coords_p_bouss')[:, 1]
    return {'ux': (g('ux_bouss'), g('ux_comp')),
            'uy': (g('uy_bouss'), g('uy_comp')),
            'theta': ((Tb - (1 - eps / 2)) / eps, (Tc - (1 - eps / 2)) / eps),
            'p': (-deramp_y(g('p_bouss_pert_raw'), y),
                  deramp_y(g('p_comp_pert_raw'), y))}


def samples(root):
    tags = [root] if os.path.isdir(os.path.join(root, 'dataset')) else \
           [os.path.join(root, t) for t in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, t))]
    for td in tags:
        for d in sorted(glob.glob(os.path.join(td, 'dataset', 'sample_*')) +
                        glob.glob(os.path.join(td, 'sample_*'))):
            mp = os.path.join(d, 'meta.json')
            if not (os.path.exists(mp) and os.path.exists(os.path.join(d, 'ux_comp.npy'))):
                continue
            m = json.load(open(mp))
            cells = int(m.get('bc_comp', {}).get('walls', {}).get('x0', {}).get('n_cells', 0))
            yield dict(dir=d, eps=round(float(m['dT']), 3), Ra=float(m['Ra']),
                       seed=int(m['seed']), flow=cells or int(m['seed']))


def fit(rs):
    e = np.log([r['eps'] for r in rs])
    out = {}
    for ch in CH:
        v = np.log([r[ch] for r in rs])
        p, cov = np.polyfit(e, v, 1, cov=True)
        resid = v - np.polyval(p, e)
        ss = ((v - v.mean()) ** 2).sum()
        out[ch] = dict(m=float(p[0]), se=float(np.sqrt(cov[0, 0])), A=float(np.exp(p[1])),
                       R2=float(1 - (resid ** 2).sum() / ss) if ss > 0 else 1.0)
    return out


def common_fit(rs, keys):
    """One exponent, one prefactor per flow (flows identified by seed)."""
    out = {}
    for ch in CH:
        y = np.log([r[ch] for r in rs])
        x = np.log([r['eps'] for r in rs])
        F = np.column_stack([np.array([1.0 if r['seed'] == f else 0.0 for r in rs]) for f in keys] + [x])
        beta, *_ = np.linalg.lstsq(F, y, rcond=None)
        res = y - F @ beta
        cov = (res @ res / (len(y) - F.shape[1])) * np.linalg.inv(F.T @ F)
        ss = ((y - y.mean()) ** 2).sum()
        out[ch] = dict(m=float(beta[-1]), se=float(np.sqrt(cov[-1, -1])),
                       A=[float(np.exp(b)) for b in beta[:-1]],
                       R2=float(1 - res @ res / ss))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--eps', default='')
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    keep = [float(s) for s in a.eps.split(',') if s]

    rows = []
    for root in a.roots:
        for r in samples(root):
            if keep and r['eps'] not in keep:
                continue
            F = load(r['dir'], r['eps'])
            for ch, (X, Y) in F.items():
                r[ch] = float(np.linalg.norm(Y - X) / np.linalg.norm(Y))
            rows.append(r)

    print('%-8s %-9s %4s  %s' % ('Ra', 'flow', 'n', '  '.join('%-17s' % c for c in CH)))
    fits = []
    for Ra in sorted(set(r['Ra'] for r in rows)):
        sub = [r for r in rows if r['Ra'] == Ra]
        keys = sorted(set(r['seed'] for r in sub))
        usable = [f for f in keys if len([r for r in sub if r['seed'] == f]) >= 2]
        for f in usable:
            rs = sorted([r for r in sub if r['seed'] == f], key=lambda z: z['eps'])
            ft = fit(rs)
            fits.append(dict(Ra=Ra, seed=f, flow=rs[0]['flow'], eps=[z['eps'] for z in rs], **ft))
            print('%-8.0f %-9d %4d  %s' % (Ra, rs[0]['flow'], len(rs), '  '.join(
                '%.3f+-%.3f' % (ft[c]['m'], ft[c]['se']) for c in CH)))
        cf = common_fit([r for r in sub if r['seed'] in usable], usable)
        print('  common slope at Ra=%.0f: %s' % (Ra, '  '.join(
            '%s=%.3f+-%.3f (A %s, R2=%.3f)' % (c, cf[c]['m'], cf[c]['se'],
                                                '/'.join('%.3f' % x for x in cf[c]['A']), cf[c]['R2'])
            for c in CH)))

    if a.json:
        out = dict(rows=rows, per_flow=fits)
        out['common'] = {}
        for Ra in sorted(set(r['Ra'] for r in rows)):
            sub = [r for r in rows if r['Ra'] == Ra]
            flows = [f for f in sorted(set(r['seed'] for r in sub))
                     if len([r for r in sub if r['seed'] == f]) >= 2]
            out['common'][str(int(Ra))] = common_fit([r for r in sub if r['seed'] in flows], flows)
        json.dump(out, open(a.json, 'w'), indent=1)
        print('saved', a.json)


if __name__ == '__main__':
    main()
