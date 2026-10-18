"""Is the per-configuration Nu denominator sane? Paired scan, all 20 configs."""
import os, re, glob
import numpy as np

DS = os.path.expanduser('~/nc_run/paired/dataset')
M = 6


def scatter(vals, cfile, n):
    c = np.load(cfile)
    ux = np.unique(np.round(c[:, 0], 12)); uy = np.unique(np.round(c[:, 1], 12))
    col = np.searchsorted(ux, np.round(c[:, 0], 12)); row = np.searchsorted(uy, np.round(c[:, 1], 12))
    img = np.full((n + 1, n + 1), np.nan); img[row, col] = vals
    return img


def nu(d, kind, eps, n=64):
    T = np.load(os.path.join(d, 'T_%s.npy' % kind)).astype(np.float64)
    th = scatter((T - (1.0 - eps / 2.0)) / eps, os.path.join(d, 'coords_T_%s.npy' % kind), n)
    xw = np.arange(M) * (1.0 / (n - 1))
    w = np.linalg.solve(np.vander(xw, M, increasing=True).T, np.eye(M)[1])
    return float(-np.nanmean(th[:, :M].dot(w)))


tags = {}
for d in sorted(glob.glob(os.path.join(DS, 'c*_Ra*'))):
    m = re.match(r'(c\d+_Ra\S+?_\d{4})_e([\d.]+)$', os.path.basename(d))
    if m and os.path.isdir(os.path.join(d, 'sample_000000')):
        tags.setdefault(m.group(1), {})[float(m.group(2))] = os.path.join(d, 'sample_000000')

print('%-18s %8s %8s %9s' % ('cfg', 'Nu_x@1.2', 'Nu_x@1.9', 'dNu@1.9'))
small = 0
for k in sorted(tags):
    d12 = tags[k].get(1.2); d19 = tags[k].get(1.9)
    x12 = nu(d12, 'bouss', 1.2); x19 = nu(d19, 'bouss', 1.9)
    r19 = (nu(d19, 'comp', 1.9) - x19) / x19
    flag = '' if abs(x19) > 1 else '  <-- |Nu|<1'
    small += abs(x19) <= 1
    print('%-18s %8.3f %8.3f %9.4f%s' % (k, x12, x19, r19, flag))
print('\nconfigs with |Nu_x(1.9)| <= 1: %d of %d' % (small, len(tags)))