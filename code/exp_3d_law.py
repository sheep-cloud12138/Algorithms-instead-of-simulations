"""3D transfer check: same R = ||Y-X||/||Y|| definition as the 2D law.

Usage: python exp_3d_law.py <batch_root> [--json out.json]
Expects <root>/<tag>/dataset/sample_*/ with the genData_3d.py output layout.
"""
import json, os, sys, glob
import numpy as np
from scipy import stats

def deramp_y(vals, y):
    A = np.column_stack([np.ones(len(y)), y])
    c, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return vals - A @ c

def load_pair(d, eps):
    g = lambda n: np.load(os.path.join(d, n + '.npy')).astype(np.float64)
    out = {}
    out['ux'] = (g('ux_bouss'), g('ux_comp'))
    out['uy'] = (g('uy_bouss'), g('uy_comp'))
    Tb, Tc = g('T_bouss'), g('T_comp')
    out['theta'] = ((Tb - (1 - eps / 2)) / eps, (Tc - (1 - eps / 2)) / eps)
    y = g('coords_p_bouss')[:, 1]
    out['p'] = (-deramp_y(g('p_bouss_pert_raw'), y), deramp_y(g('p_comp_pert_raw'), y))
    return out

def main(root):
    rows = []
    cand = [root] if os.path.isdir(os.path.join(root, 'dataset')) else [
        os.path.join(root, t) for t in sorted(os.listdir(root))]
    for td in cand:
        ds = os.path.join(td, 'dataset')
        if not os.path.isdir(ds):
            continue
        for d in sorted(glob.glob(os.path.join(ds, 'sample_*'))):
            if not os.path.exists(os.path.join(d, 'meta.json')):
                continue
            if not os.path.exists(os.path.join(d, 'ux_comp.npy')):
                continue
            m = json.load(open(os.path.join(d, 'meta.json')))
            eps = float(m['dT'])
            f = load_pair(d, eps)
            R = {}
            for ch, (X, Y) in f.items():
                D = Y - X
                R[ch] = float(np.linalg.norm(D) / np.linalg.norm(Y))
            rows.append(dict(dir=d, eps=eps, Ra=float(m['Ra']), **R))
    if not rows:
        print('no complete samples')
        return
    eps = np.array([r['eps'] for r in rows])
    print('%d samples, eps = %s' % (len(rows), sorted(set(eps))))
    for ch in ['ux', 'uy', 'theta', 'p']:
        r = np.array([x[ch] for x in rows])
        sl, ic, rv, pv, se = stats.linregress(np.log(eps), np.log(r))
        print('%-6s m=%.3f+-%.3f A=%.4f R2=%.3f' % (ch, sl, se, np.exp(ic), rv ** 2))
        for e in sorted(set(eps)):
            s = r[eps == e]
            print('        eps=%.2f n=%d median=%.4g' % (e, len(s), np.median(s)))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, 'law3d.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print('saved', out)

if __name__ == '__main__':
    main(sys.argv[1])
