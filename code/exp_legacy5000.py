"""公开 5000 组 classical-cavity（train_legacy，等温侧壁 + 绝热上下壁，eps≡1.2）上的外部验证。
分块读取，避免把两个 313 MB 张量同时展开成 float64。"""
import numpy as np, csv, os, json, torch
from scipy import stats
D0 = "/mnt/d/Desktop/算法代替仿真/research/data"
OUT = os.path.expanduser('~/nc_run/p7/legacy5000.json')
CH = ['ux', 'uy', 'theta', 'p']
res = {}
for split in ['train_legacy', 'test_legacy']:
    fx = os.path.join(D0, split, 'X.pt')
    if not os.path.exists(fx): print('skip', split); continue
    rows = list(csv.DictReader(open(os.path.join(D0, split, 'per_sample.csv'))))
    N = len(rows)
    dT = np.array([float(r['dT']) for r in rows], np.float32)
    T0 = np.array([float(r['T0']) for r in rows], np.float32)
    Ra = np.array([float(r['Ra']) for r in rows])
    eps = dT / T0
    try:
        X = torch.load(fx, weights_only=False, mmap=True)
        Y = torch.load(os.path.join(D0, split, 'Y.pt'), weights_only=False, mmap=True)
        print('mmap ok', split, tuple(X.shape))
    except Exception as e:
        print('mmap failed (%s), 普通加载' % e)
        X = torch.load(fx, weights_only=False); Y = torch.load(os.path.join(D0, split, 'Y.pt'), weights_only=False)
    assert X.shape[0] == N, (X.shape, N)
    R = np.zeros((N, 4), np.float64); umax = np.zeros(N, np.float64)
    B = 400
    for a in range(0, N, B):
        b = min(a + B, N)
        x = X[a:b].float().clone(); y = Y[a:b].float()
        for k in range(b - a):
            x[k, 2] = (x[k, 2] - (1 - dT[a + k] / (2 * T0[a + k]))) / eps[a + k]
            y[k, 2] = (y[k, 2] - (1 - dT[a + k] / (2 * T0[a + k]))) / eps[a + k]
        d = y - x
        for c in range(4):
            R[a:b, c] = (d[:, c].reshape(b - a, -1).pow(2).sum(1).sqrt()
                         / y[:, c].reshape(b - a, -1).pow(2).sum(1).sqrt()).double().numpy()
        umax[a:b] = torch.amax(torch.hypot(x[:, 0].reshape(b - a, -1), x[:, 1].reshape(b - a, -1)), 1).double().numpy()
        del x, y, d
    print('\n=== %s  n=%d  eps≡%.3f  Ra %.1e-%.1e ===' % (split, N, float(np.median(eps)), Ra.min(), Ra.max()))
    print('%-7s %-24s %-22s %s' % ('通道', 'R 中位 (p10-p90)', 'R 对 log10Ra', 'R/eps 对 umax'))
    d = dict(n=int(N), eps=float(np.median(eps)), umax_median=float(np.median(umax)))
    for k, c in enumerate(CH):
        s1, i1, r1, p1, e1 = stats.linregress(np.log10(Ra), R[:, k])
        s2, i2, r2, p2, e2 = stats.linregress(np.log10(umax), np.log10(R[:, k] / eps))
        print('%-7s %.4f (%.3f-%.3f)     r=%+.3f              a=%.3f c=%.3f r=%+.3f'
              % (c, np.median(R[:, k]), np.percentile(R[:, k], 10), np.percentile(R[:, k], 90), r1, s2, 10 ** i2, r2))
        d[c] = dict(median=float(np.median(R[:, k])), p10=float(np.percentile(R[:, k], 10)),
                    p90=float(np.percentile(R[:, k], 90)), r_logRa=float(r1),
                    umax_a=float(s2), umax_c=float(10 ** i2), umax_r=float(r2),   # 底数是 10，不是 e
                    umax_R2=float(r2 ** 2))
    res[split] = d
json.dump(res, open(OUT, 'w'), indent=1)
print('\nsaved', OUT)