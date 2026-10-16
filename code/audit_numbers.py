"""一次性重算论文里那些「只在 md 里、没有脚本产出」的数字。"""
import numpy as np, json, os
from scipy import stats
B = os.path.expanduser('~/nc_run'); P = B + '/p7'
CH = ['ux', 'uy', 'theta', 'p']
z = np.load(B + '/p2/pairs600_fixedp.npz', allow_pickle=True)
X = z['X'].astype(np.float32); Y = z['Y'].astype(np.float32)
meta = json.loads(str(z['meta'])); N = len(meta)
eps = np.array([m['eps'] for m in meta]); Ra = np.array([m['Ra'] for m in meta])
D = Y - X
R = np.stack([((D[:, c].reshape(N, -1) ** 2).sum(1) ** .5 / (Y[:, c].reshape(N, -1) ** 2).sum(1) ** .5) for c in range(4)], 1)
out = {}

print('=== eps_law.json（论文引用的 p/A） ===')
e = json.load(open(P + '/eps_law.json'))
print(json.dumps(e, indent=1)[:900])

print('\n=== §5.7 本机那行 r(R, log10Ra)：全体 vs 同表切片 eps∈[1.1,1.3] ===')
for tag, m in [('all600', np.ones(N, bool)), ('eps1.1-1.3', (eps >= 1.1) & (eps <= 1.3))]:
    vals = [stats.linregress(np.log10(Ra[m]), R[m, k])[2] for k in range(4)]
    print('  %-12s n=%-4d ' % (tag, m.sum()) + '  '.join('%s r=%+.3f' % (CH[k], vals[k]) for k in range(4)))
    out['r_%s' % tag] = [float(v) for v in vals]
# 用 R/eps 也报一遍
vals = [stats.linregress(np.log10(Ra), (R[:, k] / eps))[2] for k in range(4)]
print('  %-12s        ' % 'R/eps all600' + '  '.join('%s r=%+.3f' % (CH[k], vals[k]) for k in range(4)))

print('\n=== §5.1 BC 族哑变量：R² 与调整 R² ===')
fam = {}
for m in meta:
    fam.setdefault(tuple(m['nT']), len(fam))
f = np.array([fam[tuple(m['nT'])] for m in meta])
print('  不同 nT 族数 = %d（哑变量 %d 个）' % (len(fam), len(fam) - 1))
nn = N
for k in range(4):
    y = np.log(R[:, k])
    A0 = np.column_stack([np.ones(nn), np.log(eps)])
    b0, *_ = np.linalg.lstsq(A0, y, rcond=None)
    r2a = 1 - ((y - A0 @ b0) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    A1 = np.column_stack([A0, np.eye(len(fam))[f][:, 1:]])
    b1, *_ = np.linalg.lstsq(A1, y, rcond=None)
    r2b = 1 - ((y - A1 @ b1) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    p1, p2 = A0.shape[1], A1.shape[1]
    adj = lambda r2, p: 1 - (1 - r2) * (nn - 1) / (nn - p - 1)
    print('  %-6s R² %.3f -> %.3f   adjR² %.3f -> %.3f  (Δadj %+.3f)' % (CH[k], r2a, r2b, adj(r2a, p1), adj(r2b, p2), adj(r2b, p2) - adj(r2a, p1)))
    out['bc_%s' % CH[k]] = dict(r2a=float(r2a), r2b=float(r2b), nf=len(fam))

print('\n=== ε-blind 下界（配对集）u_x 各 eps 的最优 ε-blind ===')
pf = json.load(open(P + '/paired_final.json'))
print('  best eps-blind u_x:', {e: round(pf['eps_blind'][e]['B'][0], 4) for e in pf['eps_blind']})
print('  train@1.2 only u_x:', {e: round(pf['eps_blind'][e]['A'][0], 4) for e in pf['eps_blind']})
print('  min/max over all channels & eps of "train at 1.2 only":',
      round(min(min(pf['eps_blind'][e]['A']) for e in pf['eps_blind']), 4),
      round(max(max(pf['eps_blind'][e]['A']) for e in pf['eps_blind']), 4))
print('  min/max over all channels & eps of "best eps-blind":',
      round(min(min(pf['eps_blind'][e]['B']) for e in pf['eps_blind']), 4),
      round(max(max(pf['eps_blind'][e]['B']) for e in pf['eps_blind']), 4))
json.dump(out, open(P + '/audit_numbers.json', 'w'), indent=1)
print('\nsaved', P + '/audit_numbers.json')