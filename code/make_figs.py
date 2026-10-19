"""论文插图 2-6：全部从 ~/nc_run/p7/*.json 直接生成，任何一个结果文件缺失就跳过对应的图。"""
import numpy as np, json, os, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
B = os.path.expanduser('~/nc_run'); P = B + '/p7'
CH = [r'$u_x$', r'$u_y$', r'$\theta$', r'$p$']
J = lambda n: json.load(open(os.path.join(P, n))) if os.path.exists(os.path.join(P, n)) else None
EPS_LAB = [0.4, 0.8, 1.2, 1.6, 1.9]

# ---------------- 图 2：ε-blind 下界 ----------------
d = J('paired_final.json')
if d:
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4), dpi=170)
    for k in range(4):
        a = ax.flat[k]
        dn = [d['floor']['%.1f' % e][k] for e in EPS_LAB]
        A = [d['eps_blind']['%.1f' % e]['A'][k] for e in EPS_LAB]
        Bl = [d['eps_blind']['%.1f' % e]['B'][k] for e in EPS_LAB]
        a.plot(EPS_LAB, dn, 'o-', color='0.55', lw=1.8, ms=6, label=r'do nothing ($\hat Y=X$)')
        a.plot(EPS_LAB, A, 's-', color='#c0392b', lw=1.8, ms=6, label=r'train at $\varepsilon=1.2$ only')
        a.plot(EPS_LAB, Bl, '^-', color='#2c6fbb', lw=1.8, ms=6, label='unweighted $\\varepsilon$-mean')
        a.set_xlabel(r'$\varepsilon=\Delta T/T_0$', fontsize=11)
        a.set_ylabel(r'rel. $L_2$ error   ' + CH[k], fontsize=11)
        a.grid(alpha=.25); a.set_ylim(0, None)
        if k == 0: a.legend(fontsize=8.5, loc='upper left')
    fig.suptitle(r'Single-$\varepsilon$ training leaves an unreachable floor (paired set, 20 configurations)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .96]); f = P + '/fig2_floor.png'; fig.savefig(f); print('saved', f)

# ---------------- 图 3：统一口径下的先验消融（5 个变体，3 个种子）----------------
d = J('exp_ml_final.json')
if d:
    tags = [('A_eps_blind', 'A: ' + r'$\varepsilon$ blind', '#95a5a6'),
            ('B_eps_input', 'B: ' + r'$\varepsilon$ as input', '#7f8c8d'),
            ('P1_eps_in_x_eps', r'P1: $\varepsilon$ in net, $\times\varepsilon$ out', '#2c6fbb'),
            ('P2_eps_in_x_quad', r'P2: $\times(\varepsilon+\beta\varepsilon^2)$', '#27ae60'),
            ('P3_eps_not_in_net', r'P3: $\varepsilon$ never in net, $\times\varepsilon$ out', '#c0392b')]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.9), dpi=170)
    for j, (sp, spn) in enumerate([('extrap', 'extrapolation (train ' + r'$\varepsilon\leq1.3$)'),
                                   ('interp', 'interpolation')]):
        a = ax[j]; x = np.arange(4); w = .16
        for i, (t, l, c) in enumerate(tags):
            if t not in d[sp]: continue
            m = np.array(d[sp][t]['mean']); sd = np.array(d[sp][t]['std'])
            a.bar(x + (i - 2) * w, m, w, yerr=sd, capsize=2, label=l, color=c)
        a.plot(x, d[sp]['floor'], 'k_', ms=20, mew=2, label='do nothing')
        a.set_xticks(x); a.set_xticklabels(CH, fontsize=12)
        a.set_title(spn, fontsize=11); a.grid(alpha=.2, axis='y')
        a.set_ylabel(r'rel. $L_2$ error (mean of 3 seeds)', fontsize=11)
        if j == 1: a.legend(fontsize=8.2)
    fig.tight_layout(); f = P + '/fig3_prior.png'; fig.savefig(f); print('saved', f)

# ---------------- 图 4：学习曲线 ----------------
d = J('exp_lc_final.json')
if d:
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), dpi=170)
    var_lab = [('B_eps_input', '-o', '#c0392b', r'B: $\varepsilon$ as input'),
               ('P3_eps_not_in_net', '--s', '#2c6fbb', r'P3: predict $P(X)$, $\varepsilon$ outside')]
    ns = sorted(d['curves'][var_lab[0][0]], key=int); xn = np.array([int(v) for v in ns])
    for k in range(4):
        a = ax[k]
        for var, st, col, lb in var_lab:
            m = np.array([d['curves'][var][v]['mean'][k] for v in ns])
            s = np.array([d['curves'][var][v]['std'][k] for v in ns])
            a.errorbar(xn, m, yerr=s, fmt=st, color=col, lw=1.6, ms=5, capsize=3, label=lb)
        a.axhline(d['floor'][k], color='0.55', ls=':', lw=1.4)
        a.set_xscale('log'); a.set_xticks(xn); a.set_xticklabels([str(int(v)) for v in xn]); a.set_title(CH[k], fontsize=12)
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_xlabel('training samples', fontsize=10); a.grid(alpha=.25, which='both')
        if k == 0: a.set_ylabel(r'rel. $L_2$ error', fontsize=10); a.legend(fontsize=8)
    fig.suptitle('Up to n = 200 the prior matches or beats the ' + r'$\varepsilon$' + '-input network; that network overtakes only at n = 400 (and not for ' + r'$\theta$' + ')', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .93]); f = P + '/fig4_learning.png'; fig.savefig(f); print('saved', f)

# ---------------- 图 5：律在三种流体/几何 + 独立数据集上 ----------------
d2 = J('aspect2.json'); dw = J('water.json'); dp = J('public_transfer.json')
if os.path.exists(B + '/p2/pairs600_fixedp.npz') and dw:
    z = np.load(B + '/p2/pairs600_fixedp.npz', allow_pickle=True)
    X = z['X'].astype(np.float64); Y = z['Y'].astype(np.float64)
    meta = json.loads(str(z['meta'])); N = len(meta)
    eps = np.array([m['eps'] for m in meta]); D = Y - X
    R = np.stack([np.linalg.norm(D[:, c].reshape(N, -1), axis=1) / np.linalg.norm(Y[:, c].reshape(N, -1), axis=1) for c in range(4)], 1)
    E0 = []; R0 = []
    for dd in sorted(glob.glob(B + '/eps0/dataset/sample_*')):
        mm = json.load(open(dd + '/meta.json')); e = float(mm['dT'])
        gg = [np.load(dd + '/%s_bouss.npy' % c).astype(np.float64) for c in ['ux', 'uy', 'T']]
        cb = [np.load(dd + '/coords_%s_bouss.npy' % c) for c in ['ux', 'uy', 'T']]
        hh = [np.load(dd + '/%s_comp.npy' % c).astype(np.float64) for c in ['ux', 'uy', 'T']]
        cp = [np.load(dd + '/coords_%s_comp.npy' % c) for c in ['ux', 'uy', 'T']]
        def tim(v, c):
            ux = np.unique(np.round(c[:, 0], 12)); uy = np.unique(np.round(c[:, 1], 12))
            im = np.full((len(uy), len(ux)), np.nan)
            im[np.searchsorted(uy, np.round(c[:, 1], 12)), np.searchsorted(ux, np.round(c[:, 0], 12))] = v
            if len(ux) == 129: im = im[::2, ::2]
            return im[:64, :64]
        def dr(f):
            n0 = f.shape[0]; YY, _ = np.meshgrid(np.linspace(0, 1, n0), np.linspace(0, 1, n0), indexing='ij')
            A = np.column_stack([np.ones(f.size), YY.ravel()])
            cc, *_ = np.linalg.lstsq(A, f.ravel(), rcond=None); return (f.ravel() - A @ cc).reshape(f.shape)
        xs = [tim(gg[i], cb[i]) for i in range(3)]; ys = [tim(hh[i], cp[i]) for i in range(3)]
        xs.append(-dr(tim(np.load(dd + '/p_bouss_pert_raw.npy'), np.load(dd + '/coords_p_bouss.npy'))))
        ys.append(dr(tim(np.load(dd + '/p_comp_pert_raw.npy'), np.load(dd + '/coords_p_comp.npy'))))
        xs[2] = (xs[2] - (1 - e / 2)) / e; ys[2] = (ys[2] - (1 - e / 2)) / e
        E0.append(e); R0.append([np.linalg.norm(xs[i] - ys[i]) / np.linalg.norm(ys[i]) for i in range(4)])
    E0 = np.array(E0); R0 = np.array(R0)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.6), dpi=170)
    for k in range(4):
        a = ax.flat[k]
        a.scatter(eps, R[:, k], s=9, alpha=.55, color='#2c6fbb', linewidths=0, label=r'unit square, Pr$=0.71$ (600)')
        a.scatter(E0, R0[:, k], marker='*', s=90, color='#e67e22', edgecolors='k', linewidths=.4, label=r'$\varepsilon\to0$ checks (27)')
        if d2: a.scatter(d2['eps'], np.array(d2['R'])[:, k], s=26, marker='^', facecolors='none', edgecolors='#27ae60', linewidths=1.1, label='aspect 2:1 (104)')
        if dw: a.scatter(dw['eps'], np.array(dw['R'])[:, k], s=26, marker='s', facecolors='none', edgecolors='#c0392b', linewidths=1.1, label=r'water Pr$=4.38$ (%d)' % dw['n'])
        if dp and 'external_law' in dp:
            e = dp['external_law']
            a.errorbar([1.2], [e['R_pub_median'][k]], yerr=[[e['R_pub_median'][k] - e['R_pub_p10'][k]], [e['R_pub_p90'][k] - e['R_pub_median'][k]]],
                       fmt='D', color='k', ms=6, capsize=4, lw=1.3, label='public 4000, $\\varepsilon\\equiv1.2$')
        xs = np.linspace(.017, 2.0, 100)
        if d2: a.plot(xs, d2['law'][['ux', 'uy', 'theta', 'p'][k]]['A'] * xs ** d2['law'][['ux', 'uy', 'theta', 'p'][k]]['p'], '--', color='#27ae60', lw=1.2)
        if dw: a.plot(xs, dw['law'][['ux', 'uy', 'theta', 'p'][k]]['A'] * xs ** dw['law'][['ux', 'uy', 'theta', 'p'][k]]['p'], '-', color='#c0392b', lw=1.2)
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(.015, 2.2); a.set_ylim(2e-4, 1.2)
        a.set_xlabel(r'$\varepsilon=\Delta T/T_0$', fontsize=11); a.set_ylabel(r'$\|Y-X\|/\|Y\|$   ' + CH[k], fontsize=11)
        a.grid(alpha=.25, which='both')
        if k == 0: a.legend(fontsize=8, loc='lower right')
    fig.suptitle(r'One exponent, many prefactors', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .96]); f = P + '/fig5_law_axes.png'; fig.savefig(f); print('saved', f)

# ---------------- 图 6：跨数据集迁移（同族 / 跨 BC 族）----------------
d = J('public_transfer.json'); dl = J('public_legacy_transfer.json')
if d and 'extrap' in d:
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.6), dpi=170)
    keys = [('pub_naive', 'public, target ' + r'$D$'), ('pub_prior', 'public, target ' + r'$D/\varepsilon$'),
            ('own_prior', 'ours, target ' + r'$D/\varepsilon$')]
    keys_l = [('legacy_naive', 'public legacy, target ' + r'$D$'),
              ('legacy_prior', 'public legacy, target ' + r'$D/\varepsilon$'),
              ('own_prior', 'ours, target ' + r'$D/\varepsilon$')]
    col = ['#7f8c8d', '#27ae60', '#2c6fbb']
    for i, (dd, kk, title) in enumerate([(d, keys, 'same boundary-condition family (public 4000, random walls)'),
                                         (dl, keys_l, 'different boundary-condition family (public 5000, classical cavity)')]):
        for j, (sp, spn) in enumerate([('extrap', 'our ' + r'$\varepsilon>1.3$'), ('interp', 'our full ' + r'$\varepsilon$ range')]):
            a = ax[i][j]
            if not dd or sp not in dd: a.axis('off'); continue
            x = np.arange(4); w = .26
            for q, (key, lab) in enumerate(kk):
                if key not in dd[sp]: continue
                a.bar(x + (q - 1) * w, dd[sp][key]['g'], w, label=lab, color=col[q])
            a.plot(x, dd[sp]['floor'], 'k_', ms=20, mew=2, label='do nothing')
            a.set_xticks(x); a.set_xticklabels(CH, fontsize=12)
            a.set_title(title + chr(10) + spn, fontsize=9.5)
            a.set_ylabel(r'rel. $L_2$ error', fontsize=10); a.grid(alpha=.2, axis='y')
            if i == 0 and j == 0: a.legend(fontsize=8)
    fig.suptitle('Reusing a single-' + r'$\varepsilon$' + ' data set: the prior transfers within a boundary-condition family, not across', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .95]); f = P + '/fig6_transfer.png'; fig.savefig(f); print('saved', f)
