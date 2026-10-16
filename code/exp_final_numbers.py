"""最终数字：律的残差到底由什么决定 + 求解器墙钟（写论文用）。"""
import numpy as np, json, os, glob
z = np.load(os.path.expanduser('~/nc_run/p2/pairs600_fixedp.npz'), allow_pickle=True)
X = z['X'].astype(np.float64); Y = z['Y'].astype(np.float64); meta = json.loads(str(z['meta']))
N = len(meta); eps = np.array([m['eps'] for m in meta]); Ra = np.array([m['Ra'] for m in meta])
lR = np.log10(Ra); D = Y - X
CH = ['ux','uy','theta','p']
R = np.stack([np.linalg.norm(D[:,c].reshape(N,-1),axis=1)/np.linalg.norm(Y[:,c].reshape(N,-1),axis=1) for c in range(4)],1)
# BC 族：四种壁面的模态数
fam = np.array([''.join(str(v) for v in m['nT']) for m in meta])
uf = sorted(set(fam)); print('BC 族数：%d（600 组里出现 %d 个不同的 (nL,nR,nT,nB) 组合）' % (len(uf), len(uf)))
def r2(y, yhat): return 1-((y-yhat)**2).sum()/((y-y.mean())**2).sum()
print('\n=== 律 R=A*eps^p 的残差能否被 Ra / 边界条件族解释 ===')
for k, c in enumerate(CH):
    base = np.column_stack([eps]); coef, *_ = np.linalg.lstsq(base, R[:,k], rcond=None); o = r2(R[:,k], base@coef)
    A = np.column_stack([eps, lR]); coef, *_ = np.linalg.lstsq(A, R[:,k], rcond=None); r_ra = r2(R[:,k], A@coef)
    B = np.column_stack([eps]+[ (fam == u).astype(float) for u in uf[1:] ])
    coef, *_ = np.linalg.lstsq(B, R[:,k], rcond=None); r_fam = r2(R[:,k], B@coef)
    C = np.column_stack([eps, lR]+[ (fam == u).astype(float) for u in uf[1:] ])
    coef, *_ = np.linalg.lstsq(C, R[:,k], rcond=None); r_all = r2(R[:,k], C@coef)
    print('  %-6s 只用 eps: R²=%.3f | +logRa: %.3f | +BC族(%d项): %.3f | +两者: %.3f' % (c, o, r_ra, len(uf)-1, r_fam, r_all))
print('\n=== 求解器墙钟（本机 WSL, 64^2, OMP=1, 8 并行）===')
for tag, pat in [('600 组随机壁面', '~/nc_run/batch/dataset/*/meta.json'),
                 ('配对集 100', '~/nc_run/paired/dataset/*/sample_000000/meta.json'),
                 ('eps→0 27 组', '~/nc_run/eps0/dataset/*/meta.json')]:
    W = []
    for m in glob.glob(os.path.expanduser(pat)):
        d = json.load(open(m)); a = d.get('march', {})
        if 'bouss' in a and 'comp' in a: W.append(a['bouss'].get('wall_s',np.nan)+a['comp'].get('wall_s',np.nan))
    W = np.array([w for w in W if np.isfinite(w)])
    print('  %-14s n=%3d  中位 %6.1f s   均值 %6.1f s   p90 %7.1f s   最大 %7.1f s' %
          (tag, len(W), np.median(W), W.mean(), np.percentile(W,90), W.max()))