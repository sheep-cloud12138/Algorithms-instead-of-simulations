"""§5.10 核验：收敛质量分布 + 剔除劣收敛样本后重拟合 u_x 线性律是否变化。"""
import numpy as np, json, os
PACK = os.path.expanduser('~/nc_run/p2/pairs600_fixedp.npz')
OUT = os.path.expanduser('~/nc_run/p7/verify_noise.json'); CH = ['ux','uy','theta','p']

z = np.load(PACK, allow_pickle=True)
X, Y = z['X'].astype(np.float64), z['Y'].astype(np.float64)
meta = json.loads(str(z['meta']))
eps = np.array([m['eps'] for m in meta]); crel = np.array([m['crel'] for m in meta])
R = np.linalg.norm((Y - X).reshape(len(X), 4, -1), axis=2) / np.linalg.norm(Y.reshape(len(Y), 4, -1), axis=2)

res = {'n': int(len(eps)), 'crel': {'median': float(np.median(crel)), 'p90': float(np.percentile(crel, 90)),
       'max': float(crel.max()), 'n_gt_1e-4': int((crel > 1e-4).sum())}}
for c in range(4):
    a_all, b_all = np.polyfit(eps, R[:, c], 1)
    m = crel <= 1e-4
    a_cut, b_cut = np.polyfit(eps[m], R[m, c], 1)
    res[CH[c]] = {'all': [float(a_all), float(b_all)], 'cut': [float(a_cut), float(b_cut)],
                  'n_cut': int(m.sum()), 'R_median': float(np.median(R[:, c]))}
print(json.dumps(res, indent=1))
json.dump(res, open(OUT, 'w'), indent=1)
print('\nsaved', OUT)