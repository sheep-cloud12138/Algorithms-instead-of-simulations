"""律的残差能否被基流的「对流强度」 chi = |u|·|grad T| / |T| 解释（四个通道）。"""
import numpy as np, json, os
z = np.load(os.path.expanduser('~/nc_run/p2/pairs600_fixedp.npz'), allow_pickle=True)
X = z['X'].astype(np.float64); Y = z['Y'].astype(np.float64); meta = json.loads(str(z['meta']))
N = len(meta); eps = np.array([m['eps'] for m in meta]); lR = np.log10([m['Ra'] for m in meta])
D = Y - X
R = np.stack([np.linalg.norm(D[:,c].reshape(N,-1),axis=1)/np.linalg.norm(Y[:,c].reshape(N,-1),axis=1) for c in range(4)],1)
h = 1.0/63.0
def grad(f): return np.gradient(f, h, axis=0), np.gradient(f, h, axis=1)
chi = []
for i in range(N):
    gx, gy = grad(X[i,2]); k = np.sqrt(gx**2+gy**2)
    umag = np.sqrt(X[i,0]**2 + X[i,1]**2)
    chi.append(float(np.sqrt((umag*k)**2).mean()/np.sqrt((X[i,2]**2).mean())))
chi = np.array(chi)
from scipy import stats
print('chi 范围 %.3f - %.3f  中位 %.3f' % (chi.min(), chi.max(), np.median(chi)))
print('Spearman(chi, R/eps):  ' + '  '.join('%s=%.3f' % (c, stats.spearmanr(chi, R[:,k]/eps)[0]) for k,c in enumerate(['ux','uy','theta','p'])))
def r2(y, yh): return 1-((y-yh)**2).sum()/((y-y.mean())**2).sum()
print('\n=== R = eps*(a + b*chi)  vs  R = a*eps ===')
for k, c in enumerate(['ux','uy','theta','p']):
    y = R[:,k]
    A1 = eps[:,None]; c1, *_ = np.linalg.lstsq(A1, y, rcond=None)
    A2 = np.column_stack([eps, eps*chi]); c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
    A3 = np.column_stack([eps, eps*chi, lR*eps]); c3, *_ = np.linalg.lstsq(A3, y, rcond=None)
    print('  %-6s 只 eps: R²=%.3f | +chi: R²=%.3f (a=%.4f b=%.4f) | +chi+logRa: %.3f' %
          (c, r2(y, A1@c1), r2(y, A2@c2), c2[0], c2[1], r2(y, A3@c3)))
print('\n注：拟合自由度 2-3 个，样本 600，不存在过拟合问题。')