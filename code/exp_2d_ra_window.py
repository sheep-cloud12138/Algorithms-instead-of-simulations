"""2-D exponents over a fixed eps window, binned by Rayleigh number.

The 3-D check quotes window-local exponents at Ra=1e4 and 3e4, so the 2-D
comparison uses the samples within a factor of two of each value and the same
eps window.  Reads ~/nc_run/p2/pairs600_fixedp.npz.

Usage: python3 exp_2d_ra_window.py
"""
import json, os
import numpy as np

B = os.path.expanduser('~/nc_run')
z = np.load(B + '/p2/pairs600_fixedp.npz', allow_pickle=True)
X = z['X'].astype(np.float64); Y = z['Y'].astype(np.float64)
meta = json.loads(str(z['meta'])); N = len(meta)
eps = np.array([m['eps'] for m in meta]); Ra = np.array([m['Ra'] for m in meta])
D = Y - X
R = np.stack([np.linalg.norm(D[:, c].reshape(N, -1), axis=1) /
              np.linalg.norm(Y[:, c].reshape(N, -1), axis=1) for c in range(4)], 1)
CH = ['ux', 'uy', 'theta', 'p']

for lo, hi in [(5e3, 2e4), (2e4, 6e4)]:
    s = (Ra >= lo) & (Ra < hi) & (eps >= 0.4) & (eps <= 1.9)
    e = np.log(eps[s])
    out = []
    for k in range(4):
        p, cov = np.polyfit(e, np.log(R[s, k]), 1, cov=True)
        out.append('%s=%.3f+-%.3f' % (CH[k], p[0], np.sqrt(cov[0, 0])))
    print('Ra %.0e-%.0e n=%3d eps %.2f-%.2f  %s' % (lo, hi, s.sum(), eps[s].min(), eps[s].max(), '  '.join(out)))
