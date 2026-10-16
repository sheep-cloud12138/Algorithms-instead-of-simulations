"""Pr = 4.38（20 C 的水）几何族：打包 + 破缺律核验 + 与同配置空气样本的严格配对。

水这批用与主数据集 **完全相同** 的 --seed0 / id 0..103，只把 Pr 从 0.71 换成 4.38，
所以每个水样本都能在 600 组里找到 BC 与 Ra 逐位相同的空气样本 —— 配对比较成立。
理论预期：指数 p 不变（律的几何/流体无关性），只有系数 A 变。
"""
import numpy as np, json, os, glob
from scipy import stats
B = os.path.expanduser('~/nc_run'); DS = B + '/water/dataset'
CH = ['ux', 'uy', 'theta', 'p']

def toimg(v, c, n=64):
    ux = np.unique(np.round(c[:, 0], 12)); uy = np.unique(np.round(c[:, 1], 12))
    img = np.full((len(uy), len(ux)), np.nan)
    img[np.searchsorted(uy, np.round(c[:, 1], 12)), np.searchsorted(ux, np.round(c[:, 0], 12))] = v
    if len(ux) == 129: img = img[::2, ::2]
    return img[:n, :n]

def deramp(f):
    n0, n1 = f.shape
    Y, _ = np.meshgrid(np.linspace(0, 1, n0), np.linspace(0, 1, n1), indexing='ij')
    A = np.column_stack([np.ones(f.size), Y.ravel()])
    cc, *_ = np.linalg.lstsq(A, f.ravel(), rcond=None)
    return (f.ravel() - A @ cc).reshape(f.shape)

rows = []
for d in sorted(glob.glob(DS + '/sample_*')):
    if not os.path.exists(d + '/meta.json'): continue
    m = json.load(open(d + '/meta.json')); e = float(m['dT'])
    if not (m['march']['comp'].get('ok') and m['march']['bouss'].get('ok')): continue
    g = lambda c, k: np.load(os.path.join(d, '%s_%s.npy' % (c, k))).astype(np.float64)
    cc = lambda c, k: np.load(os.path.join(d, 'coords_%s_%s.npy' % (c, k)))
    x = [toimg(g(c, 'bouss'), cc(c, 'bouss')) for c in ['ux', 'uy', 'T']]
    y = [toimg(g(c, 'comp'), cc(c, 'comp')) for c in ['ux', 'uy', 'T']]
    pb = toimg(np.load(d + '/p_bouss_pert_raw.npy'), np.load(d + '/coords_p_bouss.npy'))
    pc = toimg(np.load(d + '/p_comp_pert_raw.npy'), np.load(d + '/coords_p_comp.npy'))
    x.append(-deramp(pb)); y.append(deramp(pc))
    x[2] = (x[2] - (1 - e / 2)) / e; y[2] = (y[2] - (1 - e / 2)) / e
    rows.append(dict(id=int(os.path.basename(d).split('_')[-1]), e=e, Ra=float(m['Ra']),
                     nT=[int(m['numT_left']), int(m['numT_right']), int(m['numT_top']), int(m['numT_bot'])],
                     rel=float(m['march']['comp'].get('achieved_rel_change', float('nan'))),
                     R=[float(np.linalg.norm(x[i] - y[i]) / np.linalg.norm(y[i])) for i in range(4)]))
print('水（Pr=4.38）可用样本 %d 组' % len(rows))
if not rows:
    raise SystemExit('no samples yet')

E = np.array([r['e'] for r in rows]); R = np.array([r['R'] for r in rows]); RA = np.array([r['Ra'] for r in rows])
print('ε ∈ [%.2f, %.2f]，Ra ∈ [%.2e, %.2e]' % (E.min(), E.max(), RA.min(), RA.max()))
print('可压缩成员 rel_change：中位 %.2e，最大 %.2e' % (np.nanmedian([r['rel'] for r in rows]), np.nanmax([r['rel'] for r in rows])))

ref = {'ux': (1.082, 0.1468), 'uy': (1.001, 0.1934), 'theta': (1.216, 0.0097), 'p': (1.121, 0.1482)}
print('\n=== Pr=4.38 里的幂律（对照：单位方腔 Pr=0.71 的 600 组）===')
print('%-7s %-40s %s' % ('通道', '水 Pr=4.38   R=A*eps^p', '空气 Pr=0.71'))
law = {}
for k, c in enumerate(CH):
    sl, ic, rv, pv, se = stats.linregress(np.log(E), np.log(R[:, k]))
    d = max(abs(sl - ref[c][0]), 2 * se)
    print('%-7s p=%.3f±%.3f  A=%.4f  R²=%.3f   %s    p=%.3f  A=%.4f'
          % (c, sl, se, np.exp(ic), rv ** 2, '一致' if abs(sl - ref[c][0]) < 4 * max(se, 0.05) else '偏离', ref[c][0], ref[c][1]))
    law[c] = dict(p=float(sl), p_se=float(se), A=float(np.exp(ic)), R2=float(rv ** 2))

print('\n=== R/eps 是否还随 Ra 走（水的次要趋势）===')
ra_tr = {}
for k, c in enumerate(CH):
    sl, ic, rv, pv, se = stats.linregress(np.log10(RA), R[:, k] / E)
    print('  %-6s slope=%+.4f  r=%+.3f  p=%.2g' % (c, sl, rv, pv))
    ra_tr[c] = dict(slope=float(sl), r=float(rv), p=float(pv))

# ---- 严格配对：水的样本 i 与 600 组里 BC/Ra 逐位相同的空气样本 ----
z = np.load(B + '/p2/pairs600_fixedp.npz', allow_pickle=True)
meta = json.loads(str(z['meta']))
Xa = z['X'].astype(np.float64); Ya = z['Y'].astype(np.float64)
Na = len(meta)
Ra_a = np.array([m['Ra'] for m in meta]); eps_a = np.array([m['eps'] for m in meta])
nT_a = [list(map(int, m['nT'])) for m in meta]
Da = Ya - Xa
Ra_own = np.stack([np.linalg.norm(Da[:, c].reshape(Na, -1), axis=1) / np.linalg.norm(Ya[:, c].reshape(Na, -1), axis=1) for c in range(4)], 1)
pair = []
for i, r in enumerate(rows):
    cand = [j for j in range(Na)
            if abs(Ra_a[j] - r['Ra']) <= 1e-9 * max(1.0, abs(r['Ra'])) and nT_a[j] == r['nT']
            and abs(eps_a[j] - r['e']) < 1e-12]
    if len(cand) == 1: pair.append((i, cand[0]))
print('\n=== 同 BC、同 Ra、同 eps，只有 Pr 不同：配对 %d / %d 组 ===' % (len(pair), len(rows)))
pr = {}
if pair:
    pi = np.array([p[0] for p in pair]); pj = np.array([p[1] for p in pair])
    print('%-7s %-28s %-28s %s' % ('通道', '水 R（中位）', '空气 R（中位）', '比值 R_水/R_空气'))
    for k, c in enumerate(CH):
        rw = R[pi, k]; ra = Ra_own[pj, k]
        rat = rw / ra
        print('%-7s %.4f                       %.4f                       %.3f  (p10 %.2f p90 %.2f)'
              % (c, np.median(rw), np.median(ra), np.median(rat), np.percentile(rat, 10), np.percentile(rat, 90)))
        pr[c] = dict(R_water_median=float(np.median(rw)), R_air_median=float(np.median(ra)),
                     ratio_median=float(np.median(rat)),
                     ratio_p10=float(np.percentile(rat, 10)), ratio_p90=float(np.percentile(rat, 90)))
    # 配对幂律：log R_water 对 log eps
    print('\n配对样本上的幂律（同一批 BC/Ra，只变 Pr）')
    for k, c in enumerate(CH):
        sl, ic, rv, pv, se = stats.linregress(np.log(E[pi]), np.log(R[pi, k]))
        sl2, ic2, rv2, pv2, se2 = stats.linregress(np.log(E[pi]), np.log(Ra_own[pj, k]))
        print('  %-6s 水 p=%.3f±%.3f | 空气 p=%.3f±%.3f' % (c, sl, se, sl2, se2))

json.dump(dict(n=len(rows), Pr=4.38, law=law, ra_trend=ra_tr, eps=E.tolist(), Ra=RA.tolist(), R=R.tolist(),
               n_paired=len(pair), paired=pr,
               paired_R_water=R[pi].tolist() if pair else [],
               paired_R_air=Ra_own[pj].tolist() if pair else [],
               paired_eps=E[pi].tolist() if pair else []), open(B + '/p7/water.json', 'w'), indent=1)
print('\nsaved ~/nc_run/p7/water.json')