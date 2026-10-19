"""Merge the 3-D sets into one file with a common flow label (cell count)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEED2CELL = {20265261: 3, 20275268: 2, 20285275: 1}
rows = []
old = json.load(open(os.path.join(ROOT, '论文', 'law3d.json'), encoding='utf-8'))
for r in old:
    if r['flow'] not in SEED2CELL:
        continue
    rows.append(dict(eps=r['eps'], Ra=r['Ra'], flow=SEED2CELL[r['flow']], seed=r['flow'],
                     **{c: r[c] for c in ('ux', 'uy', 'theta', 'p')}))
for name in ('law3d_ra3e4.json', 'law3d_ra1e4_matched.json'):
    p = os.path.join(ROOT, '论文', name)
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding='utf-8'))
    for r in d['rows']:
        rows.append(dict(eps=r['eps'], Ra=r['Ra'], flow=int(r['flow']), seed=int(r['seed']),
                         **{c: r[c] for c in ('ux', 'uy', 'theta', 'p')}))
rows.sort(key=lambda r: (r['Ra'], r['flow'], r['eps']))
out = os.path.join(ROOT, '论文', 'law3d_2ra.json')
json.dump({'rows': rows}, open(out, 'w'), indent=1)
print('%d rows -> %s' % (len(rows), out))
print('%d at Ra=1e4, %d at Ra=3e4' % (sum(r['Ra'] < 2e4 for r in rows), sum(r['Ra'] > 2e4 for r in rows)))
