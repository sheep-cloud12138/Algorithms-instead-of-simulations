"""Figure 10: 3-D paired check at two Rayleigh numbers, from law3d_2ra.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
data = json.load(open(os.path.join(ROOT, '论文', 'law3d_2ra.json'), encoding='utf-8'))['rows']
CH = [r'$u_x$', r'$u_y$', r'$\theta$', r'$p$']
KEY = ['ux', 'uy', 'theta', 'p']
NAME = {3: '3-cell Voronoi', 2: '2-cell Voronoi', 1: 'constant wall'}
COL = {3: '#2c6fbb', 2: '#e07b39', 1: '#2e8b57'}

flows = {}
for r in data:
    flows.setdefault((int(r['Ra']), int(r['flow'])), []).append(r)
flows = {k: sorted(v, key=lambda r: r['eps']) for k, v in flows.items() if len(v) >= 3}
n_paired = sum(len(v) for v in flows.values())

fig, ax = plt.subplots(2, 2, figsize=(8.0, 6.8), dpi=240)
for k in range(4):
    a = ax[k // 2][k % 2]
    for (Ra, cell), rows in sorted(flows.items(), key=lambda kv: (-kv[0][0], -kv[0][1])):
        e = np.array([r['eps'] for r in rows])
        y = np.array([r[KEY[k]] for r in rows])
        hi = Ra > 2e4
        a.plot(e, y, '--o' if hi else '-o', color=COL[cell], ms=5.2 if hi else 6.2,
               lw=1.4, mfc='white' if hi else COL[cell], mec=COL[cell])
    ref = flows[(10000, 3)]
    e0, y0 = ref[0]['eps'], ref[0][KEY[k]] * 1.0
    a.plot([e0, 1.9], [y0, y0 * 1.9 / e0], color='0.45', ls=':', lw=1.1, zorder=0)
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xticks([0.2, 0.4, 0.8, 1.2, 1.6, 1.9])
    a.set_xticklabels(['0.2', '0.4', '0.8', '1.2', '1.6', '1.9'], fontsize=9, rotation=60)
    a.xaxis.set_minor_formatter(mt.NullFormatter())
    a.set_xlim(0.17, 2.1)
    a.set_xlabel(r'$\varepsilon=\Delta T/T_0$', fontsize=13.5)
    a.set_ylabel(r'$R$   ' + CH[k], fontsize=13.5)
    a.grid(alpha=.25, which='major')
    a.tick_params(labelsize=8.2)

handles = [Line2D([], [], color=COL[c], marker='o', lw=1.4, label='%s, $Ra=10^4$' % NAME[c]) for c in (3, 2, 1)]
handles += [Line2D([], [], color='0.35', marker='o', mfc='white', ls='--', lw=1.4, label='open symbols: $Ra=3\\times10^4$'),
            Line2D([], [], color='0.45', ls=':', lw=1.1, label='slope 1')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, framealpha=.95,
           bbox_to_anchor=(0.5, 1.005))
fig.tight_layout(pad=0.6)


out = os.path.join(ROOT, '论文', 'figs', 'fig9_3d.png')
fig.savefig(out, bbox_inches='tight', pad_inches=0.12)
print('saved', out)

