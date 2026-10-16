"""追加两张图：theta 通道的流动强度律（fig7）与水/空气严格配对（fig8）。"""
import numpy as np, json, os, csv, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import torch
B = os.path.expanduser('~/nc_run'); P = B + '/p7'
PD = "/mnt/d/Desktop/算法代替仿真/research/data"
CH = [r'$u_x$', r'$u_y$', r'$\theta$', r'$p$']
J = lambda n: json.load(open(os.path.join(P, n)))

# ---------- fig7: R_theta/eps = c * uhat^a ----------
z = np.load(B + '/p2/pairs600_fixedp.npz', allow_pickle=True)
X = z['X'].astype(np.float32); Y = z['Y'].astype(np.float32)
meta = json.loads(str(z['meta'])); eps = np.array([m['eps'] for m in meta], np.float32)
N = len(eps); D = Y - X
R = np.stack([(D[:, c].reshape(N, -1) ** 2).sum(1) ** .5 / (Y[:, c].reshape(N, -1) ** 2).sum(1) ** .5 for c in range(4)], 1)
uh = np.amax(np.hypot(X[:, 0].reshape(N, -1), X[:, 1].reshape(N, -1)), 1)
del X, Y, D
series = [('ours, random walls (600)', uh, R[:, 2] / eps, '#2c6fbb', 'o', 14)]

def pub(split, tag, col, mk, n_keep=700):
    fx = B + '/pub/%s/X.pt' % split
    if not os.path.exists(fx): fx = os.path.join(PD, split, 'X.pt')
    if not os.path.exists(fx): return None
    rows = list(csv.DictReader(open(os.path.dirname(fx) + '/per_sample.csv')))
    Xp = torch.load(fx, weights_only=False, mmap=True)
    Yp = torch.load(os.path.dirname(fx) + '/Y.pt', weights_only=False, mmap=True)
    n = min(len(rows), Xp.shape[0])
    idx = np.arange(n) if n <= n_keep else np.random.default_rng(0).choice(n, n_keep, replace=False)
    dT = np.array([float(rows[i]['dT']) for i in idx], np.float32); T0 = np.array([float(rows[i]['T0']) for i in idx], np.float32)
    e = dT / T0
    u = np.zeros(len(idx)); r = np.zeros(len(idx))
    for k, i in enumerate(idx):
        x = Xp[i].float().clone(); y = Yp[i].float()
        x[2] = (x[2] - (1 - dT[k] / (2 * T0[k]))) / e[k]; y[2] = (y[2] - (1 - dT[k] / (2 * T0[k]))) / e[k]
        u[k] = float(torch.amax(torch.hypot(x[0], x[1])))
        r[k] = float(((y[2] - x[2]) ** 2).sum() ** .5 / (y[2] ** 2).sum() ** .5) / e[k]
    return (tag, u, r, col, mk, 9)

for sp, tag, col, mk in [('train', 'public random walls (700 of 4000)', '#27ae60', '^'),
                         ('test_legacy', 'public classical cavity (200 of 5200)', '#c0392b', 's')]:
    r = pub(sp, tag, col, mk)
    if r: series.append(r)
dw = J('water.json')
if dw:
    sh = np.array([np.max(np.hypot(np.array(v).reshape(2, -1)[0], np.array(v).reshape(2, -1)[1])) for v in []]) if False else None

fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=170)
fits = []
for tag, u, r, col, mk, ms in series:
    ax.scatter(u, r, s=ms, alpha=.45, color=col, marker=mk, linewidths=0, label='%s  (n=%d)' % (tag, len(u)))
    s, i0, rv, pv, se = stats.linregress(np.log(u), np.log(r))
    xs = np.linspace(u.min(), u.max(), 50)
    ax.plot(xs, np.exp(i0) * xs ** s, '-', color=col, lw=1.7)
    fits.append((tag, s, np.exp(i0), rv ** 2))
txt = 'log-log fits (on plotted subsets)\n' + '\n'.join('%s:  a=%.2f, c=%.2f, R²=%.2f' % f for f in fits)
ax.text(.985, .03, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
        bbox=dict(fc='white', ec='0.7', alpha=.92))
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'flow strength  $\hat u = \max|u_X|$', fontsize=12)
ax.set_ylabel(r'$R_\theta/\varepsilon$', fontsize=12)
ax.grid(alpha=.25, which='both'); ax.legend(fontsize=9, loc='upper left')
ax.set_title(r'The $\theta$-channel prefactor is the flow strength:  $R_\theta/\varepsilon = c\,\hat u^{\,a}$', fontsize=11)
fig.tight_layout(); f = P + '/fig7_theta_flow.png'; fig.savefig(f); print('saved', f)

# ---------- fig8: water vs air, strictly paired ----------
if dw and dw.get('n_paired', 0) > 0:
    Rw = np.array(dw['paired_R_water']); Ra = np.array(dw['paired_R_air'])
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9), dpi=170)
    for k in range(4):
        a = ax[k]
        m = max(Rw[:, k].max(), Ra[:, k].max()) * 1.1
        a.plot([0, m], [0, m], 'k--', lw=1, alpha=.6)
        a.scatter(Ra[:, k], Rw[:, k], s=22, alpha=.7, color='#c0392b', edgecolors='none')
        med = float(np.median(Rw[:, k] / Ra[:, k]))
        a.set_title('%s   median ratio %.3f' % (CH[k], med), fontsize=11)
        a.set_xlabel(r'air  Pr$=0.71$', fontsize=10); a.grid(alpha=.25)
        if k == 0: a.set_ylabel(r'water  Pr$=4.38$', fontsize=10)
        a.set_xlim(0, m); a.set_ylim(0, m)
    fig.suptitle('Same configuration, same $\\varepsilon$, only the fluid changed (104 paired samples): nothing moves', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, .92]); f = P + '/fig8_water_paired.png'; fig.savefig(f); print('saved', f)
print('extra figs done')