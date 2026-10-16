"""第二轮数字审计：把 reviewer 指出的、论文里"说法与证据不吻合"的数字全部重算一遍并落盘。"""
import numpy as np, json, os, glob
from scipy import stats
B = os.path.expanduser('~/nc_run'); P = B + '/p7'
CH = ['ux', 'uy', 'theta', 'p']
out = {}

# ---------- 1. 幂律指数 p 的窗口敏感性 ----------
z = np.load(P + '/eps_law_R.npz')
e, R = z['eps_all'], z['R_all']
print('=== 1. p 的窗口敏感性（%d 组 = 600 随机壁面 + 27 个 eps→0 核验点）===' % len(e))
out['power_windows'] = {}
for lo in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    m = e >= lo
    row = {}
    for k, c in enumerate(CH):
        p, i0, rv, pv, se = stats.linregress(np.log(e[m]), np.log(R[m, k]))
        row[c] = dict(n=int(m.sum()), p=float(p), se=float(se), A=float(np.exp(i0)), R2=float(rv ** 2))
    out['power_windows']['eps>=%.1f' % lo] = row
    if lo in (0.0, 0.8):
        print('  ε≥%.1f  n=%3d  ' % (lo, m.sum()) + '  '.join('%s p=%.3f±%.3f' % (c, row[c]['p'], row[c]['se']) for c in CH))
# 端点斜率：最大/最小 ε 处的中位 R 之比（不依赖拟合形式）
print('  端点斜率（用各 ε 上的中位 R，最小 ε → 最大 ε）：')
key = np.round(e, 3); ep = np.unique(key); med = np.array([[np.median(R[key == x, k]) for k in range(4)] for x in ep])
slo = np.log(med[-1] / med[0]) / np.log(ep[-1] / ep[0])
print('    ε %.3f → %.3f  ' % (ep[0], ep[-1]) + '  '.join('%s %.3f' % (c, slo[k]) for k, c in enumerate(CH)))
out['endpoint_slope'] = {c: float(slo[k]) for k, c in enumerate(CH)}
# 二次项：R = a eps + c eps² + b  ->  kappa = c/a
q = json.load(open(P + '/eps_law.json'))
out['quadratic_kappa'] = {}
print('  二次展开 R = a·ε(1+κ ε) + b：')
for c in CH:
    a, cc, b, r2 = q[c]['quad']
    out['quadratic_kappa'][c] = dict(a=a, c=cc, b=b, R2=r2, kappa=cc / a)
    print('    %-6s a=%.4f c=%+.4f  κ=c/a=%.2f  R²=%.3f' % (c, a, cc, cc / a, r2))

# ---------- 2. §5.2 压力通道对的约定 ----------
print('\n=== 2. 压力通道 |X_p|/|Y_p| 与 cos（三种约定都给，免得口径不明）===')
def pstats(X, Y, tag):
    A = X[:, 3].reshape(len(X), -1); Bv = Y[:, 3].reshape(len(Y), -1)
    nA = A.norm(dim=1); nB = Bv.norm(dim=1)
    r = (nA / nB).numpy(); cs = ((A * Bv).sum(1) / (nA * nB)).numpy()
    d = dict(n=len(X), ratio_global=float(nA.norm() / nB.norm()), ratio_median=float(np.median(r)),
             cos_global=float((A * Bv).sum() / (nA.norm() * nB.norm())), cos_median=float(np.median(cs)))
    print('  %-26s n=%4d  |X|/|Y| 全局 %.3f / 中位 %.3f   cos 全局 %+.3f / 中位 %+.3f'
          % (tag, d['n'], d['ratio_global'], d['ratio_median'], d['cos_global'], d['cos_median']))
    return d
import torch
out['pressure'] = {}
for tag, path, key in [('ours, 修正前 pairs600.npz', B + '/p2/pairs600.npz', 'X'),
                       ('ours, 修正后 pairs600_fixedp.npz', B + '/p2/pairs600_fixedp.npz', 'X'),
                       ('public train (4000)', B + '/pub/train/X.pt', None)]:
    if 'pdf' in tag: continue
    if not os.path.exists(path):
        print('  [skip] %s 不存在' % path); continue
    if path.endswith('.npz'):
        zz = np.load(path, allow_pickle=True)
        Xt = torch.tensor(zz['X']).float(); Yt = torch.tensor(zz['Y']).float()
    else:
        Xt = torch.load(path, weights_only=False, mmap=True).float()
        Yt = torch.load(os.path.join(os.path.dirname(path), 'Y.pt'), weights_only=False, mmap=True).float()
    out['pressure'][tag] = pstats(Xt, Yt, tag)
    del Xt, Yt

# ---------- 3. û 中位数（分块读，省内存）----------
print('\n=== 3. û = max|u_X| 的中位数（分块 mmap，逐块释放）===')
def uhat_median(path, blk=500):
    """u_hat = max sqrt(ux^2+uy^2)，与 exp_uhat_law.py 同定义。"""
    T = torch.load(path, weights_only=False, mmap=True)
    vals = []
    for k in range(0, len(T), blk):
        u = T[k:k + blk, :2].float().numpy()
        vals.append(np.hypot(u[:, 0], u[:, 1]).reshape(u.shape[0], -1).max(1))
    med = float(np.median(np.concatenate(vals)))
    del T
    return med
zz = np.load(B + '/p2/pairs600_fixedp.npz', allow_pickle=True)
u_ours = float(np.median(np.hypot(zz['X'][:, 0], zz['X'][:, 1]).reshape(len(zz['X']), -1).max(1)))   # û = max sqrt(ux²+uy²)
u = {'ours600': u_ours,
     'public_train': uhat_median(B + '/pub/train/X.pt'),
     'legacy5000': uhat_median('/mnt/d/Desktop/算法代替仿真/research/data/train_legacy/X.pt')}
for k, v in u.items(): print('  %-14s û 中位 %.4f' % (k, v))
print('  经典方腔 / 公开随机壁面 = %.0f%%   经典方腔 / 本机 = %.0f%%'
      % (100 * (u['legacy5000'] / u['public_train'] - 1), 100 * (u['legacy5000'] / u['ours600'] - 1)))
out['uhat_median'] = u
out['uhat_pct_vs_public_train'] = float(100 * (u['legacy5000'] / u['public_train'] - 1))
out['uhat_pct_vs_ours'] = float(100 * (u['legacy5000'] / u['ours600'] - 1))

# ---------- 4. 各通道 A 在 BC 族之间的比值 ----------
ps = json.load(open(P + '/public_splits.json')); lg = json.load(open(P + '/legacy5000.json'))
print('\n=== 4. legacy / train 的 R 中位之比（逐通道）===')
rat = {}
for c in CH:
    v = lg['train_legacy'][c]['median'] / ps['train'][c]['median']
    rat[c] = float(v); print('  %-6s %.2f' % (c, v))
out['ratio_legacy_over_train'] = rat

# ---------- 5. 2:1 几何的指数与单位方腔差多少 σ ----------
a2 = json.load(open(P + '/aspect2.json'))['law']
print('\n=== 5. 2:1 几何 vs 单位方腔：指数差多少个 σ ===')
sig = {}
for c in CH:
    d = a2[c]['p'] - q[c]['power'][0]
    s = np.hypot(a2[c]['p_se'], q[c]['power'][1])
    sig[c] = float(d / s); print('  %-6s 2:1 p=%.3f±%.3f  单位 p=%.3f±%.3f  差 %.1fσ'
                                 % (c, a2[c]['p'], a2[c]['p_se'], q[c]['power'][0], q[c]['power'][1], d / s))
out['aspect2_sigma'] = sig

json.dump(out, open(P + '/audit_v2.json', 'w'), indent=1)
print('\nsaved %s' % P + '/audit_v2.json')