"""Summarise the paired 3-D check from law3d.json and the solver metadata.

Usage: python exp_3d_summary.py
"""
import json
import os

import numpy as np
from scipy import stats

CH = ['ux', 'uy', 'theta', 'p']
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAW = os.path.join(ROOT, '论文', 'law3d.json')


def main():
    rows = json.load(open(LAW))
    flows = {}
    for row in rows:
        flows.setdefault(int(row['flow']), []).append(row)
    paired = {f: r for f, r in flows.items() if len({x['eps'] for x in r}) >= 3}

    out = {'n_total': len(rows), 'n_paired': sum(len(r) for r in paired.values()),
           'eps': sorted({x['eps'] for r in paired.values() for x in r}), 'channels': {}}
    for ch in CH:
        ms, ses, amps, fits = [], [], [], []
        for flow, sub in sorted(paired.items()):
            e = np.array([x['eps'] for x in sub])
            r = np.array([x[ch] for x in sub])
            sl, ic, rv, _, se = stats.linregress(np.log(e), np.log(r))
            ms.append(sl)
            ses.append(se)
            amps.append(float(np.exp(ic)))
            fits.append(dict(flow=flow, n=len(sub), m=float(sl), se=float(se),
                             A=float(np.exp(ic)), R2=float(rv ** 2)))
        out['channels'][ch] = dict(
            m_mean=float(np.mean(ms)), m_sd=float(np.std(ms, ddof=1)),
            m_min=float(np.min(ms)), m_max=float(np.max(ms)),
            max_dev_from_one=float(np.max(np.abs(np.array(ms) - 1.0))),
            A_min=float(np.min(amps)), A_max=float(np.max(amps)),
            per_flow=fits)

    z = np.load(os.path.expanduser('~/nc_run/p7/eps_law_R.npz'))
    e2, r2 = z['eps_all'], z['R_all']
    mask = (e2 >= min(out['eps'])) & (e2 <= max(out['eps']))
    out['two_d_same_window'] = dict(n=int(mask.sum()), eps=[float(e2[mask].min()), float(e2[mask].max())])
    for k, ch in enumerate(CH):
        sl, _, rv, _, se = stats.linregress(np.log(e2[mask]), np.log(r2[mask, k]))
        out['two_d_same_window'][ch] = dict(m=float(sl), se=float(se), R2=float(rv ** 2))

    costs = []
    for sub in paired.values():
        for row in sub:
            meta = json.load(open(os.path.join(row['dir'], 'meta.json')))
            costs.append((meta['march']['bouss']['wall_s'] + meta['march']['comp']['wall_s']) / 60.0)
    out['pair_cost_min'] = dict(min=float(np.min(costs)), median=float(np.median(costs)),
                                max=float(np.max(costs)), n=len(costs))

    print('%d total, %d paired, eps=%s' % (out['n_total'], out['n_paired'], out['eps']))
    for ch in CH:
        c = out['channels'][ch]
        print('%-5s m=%.3f+-%.3f [%.3f, %.3f] maxdev=%.3f A=[%.3f, %.3f]'
              % (ch, c['m_mean'], c['m_sd'], c['m_min'], c['m_max'], c['max_dev_from_one'],
                 c['A_min'], c['A_max']))
        print('      per-flow ' + ' / '.join('%.3f' % x['m'] for x in c['per_flow']))
    tw = out['two_d_same_window']
    print('2D same window n=%d: ' % tw['n'] + '  '.join('%s %.3f+-%.3f' % (ch, tw[ch]['m'], tw[ch]['se']) for ch in CH))
    print('pair cost min/median/max = %.1f / %.1f / %.1f min (n=%d)'
          % (out['pair_cost_min']['min'], out['pair_cost_min']['median'],
             out['pair_cost_min']['max'], out['pair_cost_min']['n']))
    outpath = os.path.join(ROOT, '论文', '3d_summary.json')
    json.dump(out, open(outpath, 'w'), indent=1)
    print('saved', outpath)


if __name__ == '__main__':
    main()
