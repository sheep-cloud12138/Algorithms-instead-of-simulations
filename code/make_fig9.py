"""Figure 10 of the IJHMT manuscript: 3-D paired check, rebuilt from law3d.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
data = json.load(open(os.path.join(ROOT, '论文', 'law3d.json'), encoding='utf-8'))
CH = [r'$u_x$', r'$u_y$', r'$\theta$', r'$p$']
KEY = ['ux', 'uy', 'theta', 'p']
FLOW = ['3-cell Voronoi', '2-cell Voronoi', 'constant wall']
COL = ['#2c6fbb', '#e07b39', '#2e8b57']

flows = {}
for r in data:
    flows.setdefault(r['dir'].split('sample_')[1], []).append(r)

fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.9), dpi=240)
for k in range(4):
    a = ax[k // 2][k % 2]
    for j, key in enumerate(sorted(flows)):
        rows = sorted(flows[key], key=lambda r: r['eps'])
        e = np.array([r['eps'] for r in rows])
        y = np.array([r[KEY[k]] for r in rows])
        a.plot(e, y, 'o-', color=COL[j], ms=6, lw=1.5,
               label=FLOW[int(key) % 3])
        a.plot(e, y[0] * (e / e[0]), 'k--', lw=1, alpha=.75)
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xticks([0.2, 0.4, 0.8, 1.2])
    a.set_xticklabels(['0.2', '0.4', '0.8', '1.2'])
    a.xaxis.set_minor_formatter(mt.NullFormatter())
    a.set_xlabel(r'$\varepsilon=\Delta T/T_0$', fontsize=12)
    a.set_ylabel(r'$\|Y-X\|/\|Y\|$   ' + CH[k], fontsize=12)
    a.grid(alpha=.25, which='major')
    a.tick_params(labelsize=8.5)
    if k == 0:
        a.legend(fontsize=9.5, loc='upper left', framealpha=.9)
fig.suptitle(r'3-D check: $R$ grows linearly with $\varepsilon$ in every flow (12 paired samples, $Ra=10^4$)',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .96], w_pad=1.8)
out = os.path.join(ROOT, '论文', 'figs', 'fig9_3d.png')
fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
print('saved', out)
