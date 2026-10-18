import numpy as np, json, os
from scipy import stats
z = np.load(os.path.expanduser('~/nc_run/p2/pairs600_fixedp.npz'), allow_pickle=True)
X = z['X'].astype(np.float64); Y = z['Y'].astype(np.float64); meta = json.loads(str(z['meta']))
N = len(meta); eps = np.array([m['eps'] for m in meta]); D = Y - X
CH = ['ux', 'uy', 'theta', 'p']
R = np.stack([np.linalg.norm(D[:, c].reshape(N, -1), axis=1) /
              np.linalg.norm(Y[:, c].reshape(N, -1), axis=1) for c in range(4)], 1)
fam = [tuple(int(v) for v in m['nT']) for m in meta]
uf = {f: i for i, f in enumerate(sorted(set(fam)))}
cid = np.array([uf[f] for f in fam])
rng = np.random.default_rng(0); B = 2000; out = {}
for k, c in enumerate(CH):
    y = R[:, k]; m0 = stats.linregress(np.log(eps), np.log(y)).slope
    cl, iid = [], []
    for _ in range(B):
        pick = rng.choice(len(uf), len(uf), replace=True)
        idx = np.concatenate([np.where(cid == p)[0] for p in pick])
        cl.append(stats.linregress(np.log(eps[idx]), np.log(y[idx])).slope)
        j = rng.integers(0, N, N)
        iid.append(stats.linregress(np.log(eps[j]), np.log(y[j])).slope)
    out[c] = dict(m=float(m0), cluster=list(map(float, np.percentile(cl, [2.5, 97.5]))),
                  iid=list(map(float, np.percentile(iid, [2.5, 97.5]))), n_families=len(uf))
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '论文', 'cluster_boot.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))