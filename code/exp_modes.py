"""修正场的 ε-幂次结构检验（配对集 20 配置 x 5 个 ε）：
  D(ε) = Y-X 能否写成 eps*P(X) + eps^2*Q(X) + ...？逐配置做 1/2/3 阶展开，
  再看各阶模态形状跨配置是否一致（余弦相似度）。"""
import numpy as np, os, glob, re
DS = os.path.expanduser('~/nc_run/paired/dataset')
EPS = [0.4, 0.8, 1.2, 1.6, 1.9]; CH = ['ux','uy','theta','p']

def toimg(v, c):
    ux = np.unique(np.round(c[:,0],12)); uy = np.unique(np.round(c[:,1],12))
    img = np.full((len(uy), len(ux)), np.nan)
    img[np.searchsorted(uy, np.round(c[:,1],12)), np.searchsorted(ux, np.round(c[:,0],12))] = v
    if len(ux) == 129: img = img[::2, ::2]
    return img[:64, :64]

def deramp(f):
    n = f.shape[0]
    A = np.column_stack([np.ones(n*n), np.repeat(np.linspace(0,1,n), n)])
    cc, *_ = np.linalg.lstsq(A, f.ravel(), rcond=None)
    return (f.ravel() - A@cc).reshape(n,n)

def fields(d, e):
    o = []
    for c in ['ux','uy','T']:
        b = np.load(os.path.join(d, c+'_bouss.npy')).astype(np.float64)
        p = np.load(os.path.join(d, c+'_comp.npy')).astype(np.float64)
        cb = np.load(os.path.join(d, 'coords_%s_bouss.npy' % c)); cp = np.load(os.path.join(d, 'coords_%s_comp.npy' % c))
        o.append((toimg(b, cb), toimg(p, cp)))
    pb = toimg(np.load(d+'/p_bouss_pert_raw.npy'), np.load(d+'/coords_p_bouss.npy'))
    pc = toimg(np.load(d+'/p_comp_pert_raw.npy'),  np.load(d+'/coords_p_comp.npy'))
    o.append((-deramp(pb), deramp(pc)))
    th = lambda T: (T - (1 - e/2))/e
    return [o[0], o[1], [th(o[2][0]), th(o[2][1])], o[3]]

tags = {}
for d in glob.glob(DS+'/c*_Ra*'):
    t = os.path.basename(d); m = re.match(r'(c\d+_Ra\S+?_\d{4})_e([\d.]+)$', t)
    if m and os.path.exists(os.path.join(d,'sample_000000','meta.json')):
        tags.setdefault(m.group(1), {})[float(m.group(2))] = os.path.join(d,'sample_000000')
cfgs = sorted(k for k, v in tags.items() if len(v) == len(EPS))
print('完整配对配置 %d 组 x %d 个 eps' % (len(cfgs), len(EPS)))
nrm = lambda a: float(np.sqrt((np.array(a)**2).sum()))
print('\n=== 逐配置：D(ε) 用 ε^1..ε^k 展开解释的能量比（R²） ===')
for k, name in enumerate(CH):
    tab = {1:[],2:[],3:[]}; rat = []; qs = []
    for cfg in cfgs:
        Ds = np.array([ (fields(tags[cfg][e], e)[k][1] - fields(tags[cfg][e], e)[k][0]).ravel() for e in EPS ])
        for j in (1,2,3):
            A = np.column_stack([np.array(EPS)**i for i in range(1, j+1)])
            coef, *_ = np.linalg.lstsq(A, Ds, rcond=None)
            tab[j].append(1 - ((Ds - A@coef)**2).sum()/((Ds - Ds.mean(0))**2).sum())
        A2 = np.column_stack([np.array(EPS), np.array(EPS)**2]); coef2, *_ = np.linalg.lstsq(A2, Ds, rcond=None)
        P, Q = coef2[0], coef2[1]
        rat.append(nrm(EPS[0]**2*Q)/nrm(EPS[0]*P)); qs.append(Q/nrm(Q))
    qs = np.array(qs); cs = qs@qs.T; iu = np.triu_indices(len(qs), 1)
    print('  %-6s R²(1阶)=%.4f  R²(2阶)=%.4f  R²(3阶)=%.4f  二阶/一阶能量比(ε=0.4) 中位 %.3f  跨配置 cos(Q) 中位 %.3f' %
          (name, np.median(tab[1]), np.median(tab[2]), np.median(tab[3]), np.median(rat), np.median(cs[iu])))