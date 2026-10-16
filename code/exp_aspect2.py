"""宽高比 2:1 几何族：打包 + ε-律核验（几何外推）。
网格非方形：单元 128x64，速度节点 257x129 -> 抽稀裁剪到 64x32，T/p 节点 129x65 -> 64x32。"""
import numpy as np, json, os, glob
from scipy import stats
B = os.path.expanduser('~/nc_run'); DS = B+'/cavity2/dataset'
def toimg(v, c, crop):
    ux = np.unique(np.round(c[:,0],12)); uy = np.unique(np.round(c[:,1],12))
    img = np.full((len(uy), len(ux)), np.nan)
    img[np.searchsorted(uy,np.round(c[:,1],12)), np.searchsorted(ux,np.round(c[:,0],12))] = v
    if len(ux) == 257: img = img[::2, ::2]
    return img[:crop[0], :crop[1]]
def deramp(f):
    n0, n1 = f.shape
    Y, X = np.meshgrid(np.linspace(0,1,n0), np.linspace(0,1,n1), indexing='ij')
    A = np.column_stack([np.ones(f.size), Y.ravel()])
    cc,*_ = np.linalg.lstsq(A, f.ravel(), rcond=None)
    return (f.ravel()-A@cc).reshape(f.shape)
rows = []
for d in sorted(glob.glob(DS+'/sample_*')):
    if not os.path.exists(d+'/meta.json'): continue
    m = json.load(open(d+'/meta.json')); e = float(m['dT'])
    if not (m['march']['comp'].get('ok') and m['march']['bouss'].get('ok')): continue
    g = lambda c,k: np.load(os.path.join(d,'%s_%s.npy'%(c,k))).astype(np.float64)
    cc = lambda c,k: np.load(os.path.join(d,'coords_%s_%s.npy'%(c,k)))
    x = [toimg(g(c,'bouss'), cc(c,'bouss'), (64,32)) for c in ['ux','uy','T']]
    y = [toimg(g(c,'comp'),  cc(c,'comp'),  (64,32)) for c in ['ux','uy','T']]
    pb = toimg(np.load(d+'/p_bouss_pert_raw.npy'), np.load(d+'/coords_p_bouss.npy'), (64,32))
    pc = toimg(np.load(d+'/p_comp_pert_raw.npy'), np.load(d+'/coords_p_comp.npy'), (64,32))
    x.append(-deramp(pb)); y.append(deramp(pc))
    x[2] = (x[2]-(1-e/2))/e; y[2] = (y[2]-(1-e/2))/e
    rows.append(dict(e=e, Ra=float(m['Ra']), R=[float(np.linalg.norm(x[i]-y[i])/np.linalg.norm(y[i])) for i in range(4)],
                     rel=float(m['march']['comp'].get('achieved_rel_change', np.nan))))
print('宽高比 2:1 可用样本 %d 组' % len(rows))
E = np.array([r['e'] for r in rows]); R = np.array([r['R'] for r in rows]); RA = np.array([r['Ra'] for r in rows])
print('ε ∈ [%.2f, %.2f]，Ra ∈ [%.2e, %.2e]' % (E.min(), E.max(), RA.min(), RA.max()))
CH = ['ux','uy','theta','p']
print('\n=== 2:1 几何里的幂律（与单位方腔对照）===')
print('%-7s %-34s %-30s' % ('通道','2:1 几何  R=A*eps^p','单位方腔（600 组）'))
ref = {'ux':(1.082,0.1468),'uy':(1.001,0.1934),'theta':(1.216,0.0097),'p':(1.121,0.1482)}
out = {}
for k, c in enumerate(CH):
    sl, ic, rv, pv, se = stats.linregress(np.log(E), np.log(R[:,k]))
    print('%-7s p=%.3f±%.3f  A=%.4f  R²=%.3f      p=%.3f  A=%.4f' % (c, sl, se, np.exp(ic), rv**2, ref[c][0], ref[c][1]))
    out[c] = dict(p=float(sl), p_se=float(se), A=float(np.exp(ic)), R2=float(rv**2))
print('\n=== 与 Ra 无关？2:1 几何内 R/eps 对 logRa ===')
for k, c in enumerate(CH):
    sl, ic, rv, pv, se = stats.linregress(np.log10(RA), R[:,k]/E)
    print('  %-6s slope=%+.4f  r=%+.3f  p=%.2g' % (c, sl, rv, pv))
json.dump(dict(n=len(rows), law=out, eps=E.tolist(), Ra=RA.tolist(), R=R.tolist()),
          open(os.path.expanduser('~/nc_run/p7/aspect2.json'),'w'), indent=1)
print('\nsaved ~/nc_run/p7/aspect2.json')