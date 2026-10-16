"""θ 通道的流动强度律：R_theta/eps = c * uhat^a，统一用自然对数，七个数据集一次算清并落盘。
uhat = max|u_X|（Boussinesq 输入自带），R = ||Y-X||/||Y||（theta 通道按各自 dT/T0 归一）。"""
import numpy as np, csv, os, json, torch
from scipy import stats
B = os.path.expanduser('~/nc_run'); PD = "/mnt/d/Desktop/算法代替仿真/research/data"
OUT = B + '/p7/uhat_law.json'

def fit(u, r):
    s, i0, rv, pv, se = stats.linregress(np.log(u), np.log(r))
    return dict(a=float(s), a_se=float(se), c=float(np.exp(i0)), R2=float(rv ** 2), n=int(len(u)))

res = {}
# ---- 本机 600 组 ----
z = np.load(B + '/p2/pairs600_fixedp.npz', allow_pickle=True)
X = z['X'].astype(np.float32); Y = z['Y'].astype(np.float32)
meta = json.loads(str(z['meta'])); eps = np.array([m['eps'] for m in meta], np.float32)
N = len(eps)
D = Y - X
Rth = ((D[:, 2].reshape(N, -1) ** 2).sum(1) ** .5 / (Y[:, 2].reshape(N, -1) ** 2).sum(1) ** .5)
uh = np.amax(np.hypot(X[:, 0].reshape(N, -1), X[:, 1].reshape(N, -1)), 1)
res['ours600'] = fit(uh, Rth / eps)
print('ours 600        ', {k: round(v, 4) if isinstance(v, float) else v for k, v in res['ours600'].items()})
del X, Y, D

# ---- 本机水 104 组 ----
w = json.load(open(B + '/p7/water.json'))
Ew = np.array(w['eps']); Rw = np.array(w['R'])[:, 2]

def pub(split, base, chunk=400):
    d = os.path.join(base, split)
    rows = list(csv.DictReader(open(d + '/per_sample.csv')))
    Xp = torch.load(d + '/X.pt', weights_only=False, mmap=True)
    Yp = torch.load(d + '/Y.pt', weights_only=False, mmap=True)
    n = min(len(rows), Xp.shape[0])
    dT = np.array([float(r['dT']) for r in rows], np.float32)
    T0 = np.array([float(r['T0']) for r in rows], np.float32)
    e = dT / T0
    r = np.zeros(n); u = np.zeros(n)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        x = Xp[a:b].float().clone(); y = Yp[a:b].float().clone()
        for k in range(b - a):
            x[k, 2] = (x[k, 2] - (1 - dT[a + k] / (2 * T0[a + k]))) / e[a + k]
            y[k, 2] = (y[k, 2] - (1 - dT[a + k] / (2 * T0[a + k]))) / e[a + k]
        r[a:b] = (((y[:, 2].reshape(b - a, -1) - x[:, 2].reshape(b - a, -1)) ** 2).sum(1) ** .5
                  / (y[:, 2].reshape(b - a, -1) ** 2).sum(1) ** .5).numpy()
        u[a:b] = torch.amax(torch.hypot(x[:, 0].reshape(b - a, -1), x[:, 1].reshape(b - a, -1)), 1).numpy()
    return u, r / e

for split, base in [('train', B + '/pub'), ('test_in_distribution', B + '/pub'), ('test_ood', B + '/pub'),
                    ('train_legacy', PD), ('test_legacy', PD)]:
    u, r = pub(split, base)
    res[split] = fit(u, r)
    print('%-20s' % split, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[split].items()})

json.dump(res, open(OUT, 'w'), indent=1)
print('\nsaved', OUT)