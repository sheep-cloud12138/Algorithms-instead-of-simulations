"""经典/线性 ROM 基线：POD 基 + 线性回归（ε 外推段上最容易「看起来很强」的对手）。

修正量 D = Y - X 用训练集快照的 SVD 基（覆盖 99.9% 能量，上限 60 模态）表示，
系数由线性回归从三组特征之一预测：
  m1  参数版 ε-blind : [1, log10Ra]
  m2  参数版 ε-aware : [1, eps, log10Ra, eps*log10Ra]
  m3  线性代理（信息量与网络相同）: 上述参数 + X（Boussinesq 解）的 POD 系数
测试集与 exp_ml_final.py 完全一致，指标同为 relL2 = |Yp-Y|/|Y|。
"""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, gerr, CH

OUT = os.path.expanduser('~/nc_run/p7'); os.makedirs(OUT, exist_ok=True)
X, Y, eps, meta = load600()
D = (Y - X).numpy(); Xn = X.numpy(); Yn = Y.numpy()
Ra = np.array([m['Ra'] for m in meta]); lR = np.log10(Ra)
LAM = 1e-6

def design(mask, kind, Xc=None):
    """特征矩阵：参数列 +（m3 时）X 的 POD 系数列。"""
    n = int(mask.sum())
    cols = [np.ones(n), lR[mask]]
    if kind != 'm1': cols += [eps[mask], eps[mask] * lR[mask]]
    if kind == 'm3': cols.append(Xc[mask])
    return np.column_stack(cols)

res = {'note': 'D 的 POD 基取自训练集快照（99.9% 能量，<=60 模态）；岭回归 lam=1e-6；特征做训练集标准化'}
for split in ['extrap', 'interp']:
    tr, te = splits(eps, split)
    # X 的 POD 系数（特征用）
    Xf = Xn.transpose(0, 2, 3, 1).reshape(NX := len(eps), -1)
    Ux, Sx, Vtx = np.linalg.svd(Xf[tr], full_matrices=False)
    ex = np.cumsum(Sx ** 2) / (Sx ** 2).sum()
    kx = min(int(np.searchsorted(ex, 0.999) + 1), 40)
    Xc = (Xf - Xf[tr].mean(0)) @ Vtx[:kx].T
    out = dict(n_train=int(tr.sum()), n_test=int(te.sum()), kx=kx, floor=gerr(X[te], Y[te]), ch={})
    F = {k: design(tr, k, Xc) for k in ['m1', 'm2', 'm3']}
    Fte = {k: design(te, k, Xc) for k in ['m1', 'm2', 'm3']}
    for c in range(4):
        A = D[tr, c].reshape(tr.sum(), -1)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        en = np.cumsum(S ** 2) / (S ** 2).sum()
        k = min(int(np.searchsorted(en, 0.999) + 1), 60)
        Phi = Vt[:k].T                                  # (n_space, k)
        Coef = U[:, :k] * S[:k]                         # (n_tr, k)
        rec = {'k': k, 'energy': float(en[k - 1])}
        for kind in ['m1', 'm2', 'm3']:
            Atr, Ate = F[kind], Fte[kind]
            mu, sd = Atr.mean(0), Atr.std(0) + 1e-12
            Atr = (Atr - mu) / sd; Ate = (Ate - mu) / sd
            W = np.linalg.solve(Atr.T @ Atr + LAM * np.eye(Atr.shape[1]), Atr.T @ Coef)
            Dp = Ate @ W @ Phi.T
            Yp = Xn[te, c].reshape(te.sum(), -1) + Dp
            Yt = Yn[te, c].reshape(te.sum(), -1)
            rec[kind] = float(np.linalg.norm(Yp - Yt) / np.linalg.norm(Yt))
        out['ch'][CH[c]] = rec
        print('%-7s %-6s k=%2d  m1=%.4f  m2=%.4f  m3=%.4f' %
              (split, CH[c], k, rec['m1'], rec['m2'], rec['m3']), flush=True)
    res[split] = out
json.dump(res, open(os.path.join(OUT, 'pod_rom.json'), 'w'), indent=1)
print('saved ~/nc_run/p7/pod_rom.json')