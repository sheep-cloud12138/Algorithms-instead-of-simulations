"""Integral Nu gap on the controlled epsilon scan (20 configurations x 5 contrasts)."""
import json, os, re, glob
import numpy as np

DS = os.path.expanduser('~/nc_run/paired/dataset')
M = 6
def scatter(vals, cfile, N):
    c = np.load(cfile); ux = np.unique(np.round(c[:, 0], 12)); uy = np.unique(np.round(c[:, 1], 12))
    col = np.searchsorted(ux, np.round(c[:, 0], 12)); row = np.searchsorted(uy, np.round(c[:, 1], 12))
    img = np.full((N, N), np.nan); img[row, col] = vals; return img

def theta(T, eps):
    return (T - (1.0 - eps / 2.0)) / eps

tags = {}
for d in sorted(glob.glob(os.path.join(DS, 'c*_Ra*'))):
    m = re.match(r'(c\d+_Ra\S+?_\d{4})_e([\d.]+)$', os.path.basename(d))
    if m and os.path.isdir(os.path.join(d, 'sample_000000')):
        tags.setdefault(m.group(1), {})[float(m.group(2))] = os.path.join(d, 'sample_000000')
cfg = [k for k in sorted(tags) if len(tags[k]) == 5]
dx = 1.0 / 64.0; xw = np.arange(M) * dx
w = np.linalg.solve(np.vander(xw, M, increasing=True).T, np.eye(M)[1])
rows = []
for k in cfg:
    for e in [0.4, 0.8, 1.2, 1.6, 1.9]:
        d = tags[k][e]
        out = {}
        for kind in ('bouss', 'comp'):
            T = np.load(os.path.join(d, 'T_%s.npy' % kind)).astype(np.float64)
            c = np.load(os.path.join(d, 'coords_T_%s.npy' % kind))
            th = scatter(theta(T, e), os.path.join(d, 'coords_T_%s.npy' % kind), 65)
            out[kind] = float(-th[:, :M].dot(w).mean())
        rows.append(dict(cfg=k, eps=e, nu_x=out['bouss'], nu_y=out['comp'],
                         rel=(out['comp'] - out['bouss']) / out['bouss']))
res = {}
for e in [0.4, 0.8, 1.2, 1.6, 1.9]:
    s = np.array([r['rel'] for r in rows if r['eps'] == e])
    res[str(e)] = dict(n=len(s), mean=float(s.mean()), median=float(np.median(s)),
                       p10=float(np.percentile(s, 10)), p90=float(np.percentile(s, 90)),
                       pos=float((s > 0).mean()))
    print('eps %.1f  mean %+.4f  median %+.4f  p10 %+.4f  p90 %+.4f  frac>0 %.2f' %
          (e, s.mean(), np.median(s), np.percentile(s, 10), np.percentile(s, 90), (s > 0).mean()))
E = np.array([0.4, 0.8, 1.2, 1.6, 1.9]); Rm = np.array([res[str(e)]['mean'] for e in E])
sl, ic = np.polyfit(E, Rm, 1)
print('linear fit dNu/Nu = %.4f eps %+.4f' % (sl, ic))
print('ratio to eps: %s' % ['%.4f' % (v / e) for v, e in zip(Rm, E)])
json.dump(dict(n_cfg=len(cfg), stencil=M, by_eps=res,
               fit=dict(slope=float(sl), intercept=float(ic))), open('论文/dnu_eps.json', 'w'), indent=1)
print('saved 论文/dnu_eps.json')
