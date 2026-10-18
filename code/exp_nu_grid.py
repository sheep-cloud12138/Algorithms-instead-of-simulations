"""Grid convergence of the wall Nusselt number on the existing 64^2 / 128^2 pairs."""
import os, json
import numpy as np

R = os.path.expanduser('~/nc_run')
M = 6
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'nu_grid.json')

CASES = [
    ('c08_Ra1e4_1100', e, os.path.join(R, 'paired', 'dataset', '%s_e%g' % ('c08_Ra1e4_1100', e)),
     os.path.join(R, 'grid128', 'dataset', '%s_e%g' % ('c08_Ra1e4_1100', e)))
    for e in (0.4, 0.8, 1.2, 1.6, 1.9)
] + [
    ('c16_Ra1e6_1100', 0.4, os.path.join(R, 'paired', 'dataset', 'c16_Ra1e6_1100_e0.4'),
     os.path.join(R, 'grid_ra1e6', 'd128_e0.4')),
    ('c16_Ra1e6_1100', 1.2, os.path.join(R, 'paired', 'dataset', 'c16_Ra1e6_1100_e1.2'),
     os.path.join(R, 'grid_ra1e6', 'd128_e1.2')),
    ('c16_Ra1e6_1100', 1.9, os.path.join(R, 'paired', 'dataset', 'c16_Ra1e6_1100_e1.9'),
     os.path.join(R, 'grid_ra1e6', 'd128')),
]


def scatter(vals, cfile, n):
    c = np.load(cfile)
    ux = np.unique(np.round(c[:, 0], 10)); uy = np.unique(np.round(c[:, 1], 10))
    col = np.searchsorted(ux, np.round(c[:, 0], 10)); row = np.searchsorted(uy, np.round(c[:, 1], 10))
    img = np.full((n + 1, n + 1), np.nan); img[row, col] = vals
    return img


def nu(d, kind, eps, n):
    T = np.load(os.path.join(d, 'T_%s.npy' % kind)).astype(np.float64)
    c = os.path.join(d, 'coords_T_%s.npy' % kind)
    th = scatter((T - (1.0 - eps / 2.0)) / eps, c, n)
    xw = np.arange(M) * (1.0 / (n - 1))
    w = np.linalg.solve(np.vander(xw, M, increasing=True).T, np.eye(M)[1])
    return float(-np.nanmean(th[:, :M].dot(w)))


rows = []
for cfg, eps, d64, d128 in CASES:
    r = dict(cfg=cfg, eps=eps)
    for n, d in ((64, d64), (128, d128)):
        try:
            x = nu(os.path.join(d, 'sample_000000'), 'bouss', eps, n)
            y = nu(os.path.join(d, 'sample_000000'), 'comp', eps, n)
        except FileNotFoundError as ex:
            print('skip', cfg, eps, ex); r = None; break
        r['nu_x%d' % n] = x; r['nu_y%d' % n] = y
        r['rel%d' % n] = (y - x) / x
    if r:
        rows.append(r)
        print('%-16s eps %.1f  Nu_x 64/128 %.5f/%.5f  dNu 64 %+.4f  128 %+.4f  change %+.4f'
              % (cfg, eps, r['nu_x64'], r['nu_x128'], r['rel64'], r['rel128'], r['rel128'] - r['rel64']))
json.dump(rows, open(OUT, 'w'), indent=1)
if rows:
    c = np.array([r['rel128'] - r['rel64'] for r in rows])
    print('\nmedian |change| %.4f   max %.4f   (dNu at 64: median %.4f)'
          % (np.median(np.abs(c)), np.abs(c).max(), np.median([r['rel64'] for r in rows])))