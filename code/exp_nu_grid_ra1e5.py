"""Nu grid convergence at Ra=1e5, eps=1.2 and 1.9, isothermal sidewalls."""
import os, json
import numpy as np

R = os.path.expanduser('~/nc_run')
M = 6
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'nu_grid_ra1e5.json')


def scatter(vals, cfile, n):
    c = np.load(cfile)
    ux = np.unique(np.round(c[:, 0], 10)); uy = np.unique(np.round(c[:, 1], 10))
    col = np.searchsorted(ux, np.round(c[:, 0], 10)); row = np.searchsorted(uy, np.round(c[:, 1], 10))
    img = np.full((n + 1, n + 1), np.nan); img[row, col] = vals
    return img


def nu(d, kind, eps, n):
    T = np.load(os.path.join(d, 'T_%s.npy' % kind)).astype(np.float64)
    th = scatter((T - (1.0 - eps / 2.0)) / eps, os.path.join(d, 'coords_T_%s.npy' % kind), n)
    xw = np.arange(M) * (1.0 / (n - 1))
    w = np.linalg.solve(np.vander(xw, M, increasing=True).T, np.eye(M)[1])
    return float(-np.nanmean(th[:, :M].dot(w)))


rows = []
for eps in (1.2, 1.9):
    cfg = 'c12_Ra1e5_1100'
    d64 = os.path.join(R, 'paired', 'dataset', cfg + '_e%.1f' % eps, 'sample_000000')
    d128 = os.path.join(R, 'grid_ra1e5', cfg + '_e%.1f' % eps, 'sample_000000')
    r = dict(cfg=cfg, eps=eps)
    for n, d in ((64, d64), (128, d128)):
        r['x%d' % n] = nu(d, 'bouss', eps, n); r['y%d' % n] = nu(d, 'comp', eps, n)
        r['rel%d' % n] = (r['y%d' % n] - r['x%d' % n]) / r['x%d' % n]
    rows.append(r)
    print('eps %.1f  Nu_x 64 %.4f 128 %.4f (+%.2f%%)  dNu 64 %+.4f 128 %+.4f  change %+.4f'
          % (eps, r['x64'], r['x128'], 100 * (r['x128'] / r['x64'] - 1), r['rel64'], r['rel128'], r['rel128'] - r['rel64']))
c = np.array([r['rel128'] - r['rel64'] for r in rows])
json.dump(dict(rows=rows, summary=dict(n=len(rows), change=float(c.mean()), max_abs_change=float(np.abs(c).max()))),
          open(OUT, 'w'), indent=1)
print('saved', OUT)
