"""公开数据集另外两个 held-out 划分：test_legacy（经典等温壁方腔）与 test_ood
（竖壁 Fourier 模态数 4-6，超出训练分布）。用来独立检验「边界条件族几乎不影响律」。

每个划分的 dT/T0 直接读它自己的 per_sample.csv，避免约定假设。
"""
import numpy as np, json, os, csv
import torch
from scipy import stats
B = os.path.expanduser('~/nc_run/pub'); P = os.path.expanduser('~/nc_run/p7')
CH = ['ux', 'uy', 'theta', 'p']
ref = {'ux': 0.1797, 'uy': 0.2435, 'theta': 0.0228, 'p': 0.1858}   # 公开 train 4000 的中位
out = {}
for split in ['train', 'test_in_distribution', 'test_legacy', 'test_ood']:
    fx, fy, fc = B + '/%s/X.pt' % split, B + '/%s/Y.pt' % split, B + '/%s/per_sample.csv' % split
    if not (os.path.exists(fx) and os.path.exists(fy) and os.path.exists(fc)):
        print('[skip] %s（文件未就绪）' % split); continue
    X = torch.load(fx, weights_only=False).float(); Y = torch.load(fy, weights_only=False).float()
    rows = list(csv.DictReader(open(fc)))
    N = X.shape[0]
    dT = np.array([float(r['dT']) for r in rows]); T0 = np.array([float(r['T0']) for r in rows])
    Ra = np.array([float(r['Ra']) for r in rows])
    eps = dT / T0
    X = X.clone(); Y = Y.clone()
    th = lambda T, k: (T - (1.0 - dT[k] / (2 * T0[k]))) / eps[k]
    for k in range(N):
        X[k, 2] = th(X[k, 2], k); Y[k, 2] = th(Y[k, 2], k)
    D = Y - X
    R = np.stack([np.linalg.norm(D[:, c].reshape(N, -1), axis=1) / np.linalg.norm(Y[:, c].reshape(N, -1), axis=1) for c in range(4)], 1)
    print('\n=== %s  (n=%d, dT/T0=%.3f, Pr=%.2f, Ra %.1e-%.1e) ===' % (split, N, float(np.median(eps)), float(rows[0]['Pr']), Ra.min(), Ra.max()))
    d = dict(n=int(N), eps=float(np.median(eps)), pr=float(rows[0]['Pr']))
    print('%-7s %-28s %-24s %s' % ('通道', 'R 中位 (p10-p90)', 'R 对 log10Ra 相关', '对照：train 4000'))
    for c in range(4):
        s, i0, rv, pv, se = stats.linregress(np.log10(Ra), R[:, c])
        print('%-7s %.4f (%.3f-%.3f)          r=%+.3f                 %.4f' %
              (CH[c], np.median(R[:, c]), np.percentile(R[:, c], 10), np.percentile(R[:, c], 90), rv, ref[CH[c]]))
        d[CH[c]] = dict(median=float(np.median(R[:, c])), p10=float(np.percentile(R[:, c], 10)),
                        p90=float(np.percentile(R[:, c], 90)), r_logRa=float(rv), p_logRa=float(pv))
    out[split] = d
json.dump(out, open(P + '/public_splits.json', 'w'), indent=1)
print('\nsaved ~/nc_run/p7/public_splits.json')