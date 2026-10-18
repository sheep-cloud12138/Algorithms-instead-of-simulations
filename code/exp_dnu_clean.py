"""dNu(eps) restricted to configurations where the wall flux is non-degenerate."""
import os, re, glob, json
import numpy as np

DS = os.path.expanduser('~/nc_run/paired/dataset')
M = 6
EPS = [0.4, 0.8, 1.2, 1.6, 1.9]


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

rows = []
for k in sorted(tags):
    r = {'cfg': k}
    for e in EPS:
        d = tags[k][e]
        x = nu(d, 'bouss', e); y = nu(d, 'comp', e)
        r['nu%d' % int(e * 10)] = x
        r['rel%d' % int(e * 10)] = (y - x) / x
    rows.append(r)

iso = [r for r in rows if r['cfg'].split('_')[-1].startswith('1100')]
big = [r for r in rows if abs(r['nu19']) > 1.0]
print("isothermal configs: %d | |Nu(1.9)|>1: %d | all: %d" % (len(iso), len(big), len(rows)))
print("\n%-7s %18s %18s %18s" % ('eps', 'all 20', 'isothermal 5', '|Nu|>1 (12)'))
out = {}
for e in EPS:
    k = 'rel%d' % int(e * 10)
    a = np.array([r[k] for r in rows]); i = np.array([r[k] for r in iso]); b = np.array([r[k] for r in big])
    out[str(e)] = dict(all=[float(a.mean()), float(np.median(a))],
                       iso=[float(i.mean()), float(np.median(i)), float(i.std(ddof=1))],
                       big=[float(b.mean()), float(np.median(b)), float(b.std(ddof=1))])
    print('%-7.1f %+8.4f (med %+.4f) %+8.4f (sd %.4f) %+8.4f (sd %.4f)'
          % (e, a.mean(), np.median(a), i.mean(), i.std(ddof=1), b.mean(), b.std(ddof=1)))
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'dnu_clean.json'), 'w'), indent=1)