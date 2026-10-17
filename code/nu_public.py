"""Wall heat transfer of the Boussinesq/compressible pair on the public classical cavity.

Reads the public test_legacy split (Menglik et al., CC BY 4.0) with a numpy-only
reader (code/pt_numpy.py), computes the Nusselt number at the hot wall from a
one-sided wall gradient, and writes JSON + figure (fig_wallflux.png).
"""
import sys, csv, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pt_numpy import load

DATA = os.path.join(HERE, '..', 'data_public')
OUT = os.path.join(HERE, '..', '论文')
DT, PR, DX, M = 1.2, 0.71, 1.0 / 63, 6

X = load(os.path.join(DATA, 'test_legacy_X.pt'))
Y = load(os.path.join(DATA, 'test_legacy_Y.pt'))
rows = list(csv.DictReader(open(os.path.join(DATA, 'test_legacy_per_sample.csv'))))
Ra = np.array([float(r['Ra']) for r in rows])

xw = np.arange(M) * DX
rhs = np.zeros(M)
rhs[1] = 1.0
w = np.linalg.solve(np.vander(xw, M, increasing=True).T, rhs)


def local_nu(T):
    th = (T[:, 2] - 1.0) / DT
    return -np.tensordot(th[:, :, :M], w, axes=([2], [0]))


lx, ly = local_nu(X), local_nu(Y)
nxm, nym = lx.mean(1), ly.mean(1)
rel = (nym - nxm) / nxm
loc = (ly - lx) / lx
L1 = np.abs(ly - lx).sum(1) / np.abs(lx).sum(1)

res = {
    'n': int(len(Ra)), 'eps': 1.2, 'stencil_points': M,
    'nu_bouss_mean': float(nxm.mean()), 'nu_comp_mean': float(nym.mean()),
    'dnu_mean': float(rel.mean()), 'dnu_ratio_of_means': float(nym.mean()/nxm.mean()-1.0), 'dnu_median': float(np.median(rel)),
    'dnu_p10': float(np.percentile(rel, 10)), 'dnu_p90': float(np.percentile(rel, 90)),
    'dnu_min': float(rel.min()), 'dnu_max': float(rel.max()),
    'local_median': float(np.median(loc)), 'local_p10': float(np.percentile(loc, 10)),
    'local_p90': float(np.percentile(loc, 90)),
    'l1_median': float(np.median(L1)), 'l1_p90': float(np.percentile(L1, 90)),
}
json.dump(res, open(os.path.join(OUT, 'nu_result.json'), 'w'), indent=1)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
y = np.linspace(0, 1, lx.shape[1])
ax[0].plot(lx.mean(0) / nxm.mean(), y, label='Boussinesq $X$', lw=1.8)
ax[0].plot(ly.mean(0) / nym.mean(), y, label='compressible $Y$', lw=1.8, ls='--')
ax[0].set_xlabel('local $Nu(y)$ / mean $Nu$')
ax[0].set_ylabel('$y$')
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title('(a) wall-flux profile, mean over 200 samples', fontsize=9)

prof = (ly - lx).mean(0)
ax[1].axhline(0, color='k', lw=0.6)
ax[1].plot(prof, y, lw=1.8, color='tab:red')
ax[1].fill_betweenx(y, 0, prof, where=prof > 0, alpha=0.2, color='tab:red', lw=0)
ax[1].fill_betweenx(y, 0, prof, where=prof < 0, alpha=0.2, color='tab:blue', lw=0)
ax[1].set_xlabel(r'$\Delta Nu(y)$ (compressible $-$ Boussinesq)')
ax[1].set_ylabel('$y$')
ax[1].set_title(r'(b) redistribution along the wall, $\varepsilon=1.2$', fontsize=9)
for a in ax:
    a.tick_params(labelsize=8)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figs', 'fig_wallflux.png'), dpi=300)
print('saved fig9')
