"""Nu grid convergence at Ra=1e6, eps=1.2 over the four configurations."""
import os, json
import numpy as np

R = os.path.expanduser('~/nc_run')
M = 6
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'nu_grid_ra1e6.json')

# Only the isothermal-sidewall (legacy 1-1-0-0) configuration belongs in a
# wall-Nu grid check. The random-wall runs are stored elsewhere and have
# |Nu_X| <= 1 in some cases, where a relative Nu change is not defined.
CASES = [
    ('c16_Ra1e6_1100', os.path.join(R, 'grid_ra1e6', 'd128_e1.2')),
]


def scatter(vals, cfile, n):
    c = np.load(cfile)
    ux = np.unique(np.round(c[:, 0], 10)); uy = np.unique(np.round(c[:, 1], 10))
    col = np.searchsorted(ux, np.round(c[:, 0], 10)); row = np.searchsorted(uy, np.round(c[:, 1], 10))
    img = np.full((n + 1, n + 1), np.nan); img[row, col] = vals
    return img


def nu(d, kind, eps, n):
    T = np.load(os.path.join(d, 'T_%s.npy' % kind)).astype(np.float64)
    th = scatter((T - (1.0 - eps / 2.0)) / eps,
                 os.path.join(d, 'coords_T_%s.npy' % kind), n)
    xw = np.arange(M) * (1.0 / (n - 1))
    w = np.linalg.solve(np.vander(xw, M, increasing=True).T, np.eye(M)[1])
    return float(-np.nanmean(th[:, :M].dot(w)))


rows = []
for cfg, d128 in CASES:
    d64 = os.path.join(R, 'paired', 'dataset', cfg + '_e1.2', 'sample_000000')
    r = dict(cfg=cfg, eps=1.2)
    for n, d in ((64, d64), (128, os.path.join(d128, 'sample_000000'))):
        r['x%d' % n] = nu(d, 'bouss', 1.2, n); r['y%d' % n] = nu(d, 'comp', 1.2, n)
        r['rel%d' % n] = (r['y%d' % n] - r['x%d' % n]) / r['x%d' % n]
    rows.append(r)
    print('%-16s Nu_x 64 %.4f 128 %.4f (+%.2f%%)  dNu 64 %+.4f 128 %+.4f  change %+.4f'
          % (cfg, r['x64'], r['x128'], 100 * (r['x128'] / r['x64'] - 1),
             r['rel64'], r['rel128'], r['rel128'] - r['rel64']))
c = np.array([r['rel128'] - r['rel64'] for r in rows])
a64 = np.array([r['rel64'] for r in rows]); a128 = np.array([r['rel128'] for r in rows])
print('\nRa=1e6 eps=1.2, n=%d' % len(rows))
print('  dNu at 64 : mean %+.4f  sd %.4f' % (a64.mean(), a64.std(ddof=1)))
print('  dNu at 128: mean %+.4f  sd %.4f' % (a128.mean(), a128.std(ddof=1)))
print('  grid change: mean %+.4f  median %+.4f  max|.| %.4f' % (c.mean(), np.median(c), np.abs(c).max()))
print('  Nu_x grid error: mean %+.2f%%' % (100 * np.mean([r['x128'] / r['x64'] - 1 for r in rows])))
summary = dict(n=len(rows), dnu64=float(a64.mean()), dnu128=float(a128.mean()),
               change=float(c.mean()))
if len(rows) > 1:
    summary.update(dnu64_sd=float(a64.std(ddof=1)), dnu128_sd=float(a128.std(ddof=1)))
json.dump(dict(rows=rows, summary=summary), open(OUT, 'w'), indent=1)