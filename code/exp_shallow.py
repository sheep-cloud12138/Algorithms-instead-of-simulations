"""浅层基线：线性岭回归与 RBF 核岭回归（KRR），输入与网络完全相同（X 的四个场 + 可选 ε），
目标 D = Y - X。用来回答「这个修正是不是必须用深度网络」，并给 §5.4 一个可复现的对手。

特征：每个通道先除以训练子集的标准差；ε 变体额外拼一列常数场 ε。
核：RBF，γ = median-heuristic（γ = 1/median‖x_i-x_j‖²）。岭参数 λ 取 1e-3（目标已按训练集尺度归一）。
"""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, gerr, CH

P = os.path.expanduser('~/nc_run/p7'); os.makedirs(P, exist_ok=True)
X, Y, eps, meta = load600()
Xf = X.numpy().reshape(len(eps), -1).astype(np.float32)
Df = (Y - X).numpy().reshape(len(eps), -1).astype(np.float32)

def design(tr, te, use_eps):
    mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-30
    A, B = (Xf[tr] - mu) / sd, (Xf[te] - mu) / sd
    if use_eps:
        A = np.column_stack([A, eps[tr]]); B = np.column_stack([B, eps[te]])
    return A, B

def ridge(A, B, T, lam=1e-3):
    """线性核岭回归，对偶形式（K = A A^T，n x n）。"""
    K = A @ A.T
    alpha = np.linalg.solve(K + lam * np.mean(np.diag(K)) * np.eye(len(A)), T)
    return (B @ A.T) @ alpha


def krr(A, B, T, lam=1e-3):
    """RBF 核岭回归，gamma 用 median heuristic。"""
    n2 = (A ** 2).sum(1)
    D2 = n2[:, None] + n2[None, :] - 2 * A @ A.T
    np.maximum(D2, 0, out=D2)
    gamma = 1.0 / max(np.median(D2), 1e-12)
    K = np.exp(-gamma * D2); del D2
    alpha = np.linalg.solve(K + lam * np.eye(len(A)), T)
    n2b = (B ** 2).sum(1)
    D2b = n2b[:, None] + n2[None, :] - 2 * B @ A.T
    np.maximum(D2b, 0, out=D2b)
    return np.exp(-gamma * D2b) @ alpha


out = {'note': 'X 逐通道按训练子集标准化；目标 D；岭 λ=1e-3（按迹归一）；RBF 核 γ=median-heuristic'}
for split in ['extrap', 'interp']:
    tr, te = splits(eps, split)
    sD = np.abs(Df[tr]).mean()
    out[split] = dict(n_train=int(tr.sum()), n_test=int(te.sum()), floor=gerr(X[te], Y[te]), models={})
    print('\n=== %s  n_train=%d n_test=%d ===' % (split, tr.sum(), te.sum()), flush=True)
    for use_eps in [False, True]:
        A, B = design(tr, te, use_eps)
        for name, fn in [('ridge', ridge), ('krr', krr)]:
            Dp = fn(A, B, Df[tr] / sD) * sD
            Yp = X[te].numpy().reshape(te.sum(), -1) + Dp
            g = gerr(__import__('torch').tensor(Yp.reshape(te.sum(), 4, 64, 64)), Y[te])
            tag = '%s_%s' % (name, 'eps' if use_eps else 'blind')
            out[split]['models'][tag] = g
            print('  %-12s %s' % (tag, ' '.join('%s=%.4f' % (CH[i], g[i]) for i in range(4))), flush=True)
json.dump(out, open(os.path.join(P, 'shallow_baselines.json'), 'w'), indent=1)
print('\nsaved %s/shallow_baselines.json' % P)