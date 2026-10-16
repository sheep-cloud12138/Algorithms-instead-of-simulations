"""eps-律的最终标定（用修正 p 通道后的数据）：
  R = |Y-X|/|Y| 的线性/二次/幂律三种形式，含 eps->0 的独立核验点。"""
import numpy as np, json, os, glob
from scipy import stats

B = os.path.expanduser('~/nc_run')
CH = ['ux','uy','theta','p']
z = np.load(B+'/p2/pairs600_fixedp.npz', allow_pickle=True)
X = z['X'].astype(np.float64); Y = z['Y'].astype(np.float64); meta = json.loads(str(z['meta']))
N = len(meta); eps = np.array([m['eps'] for m in meta]); Ra = np.array([m['Ra'] for m in meta])
D = Y - X
R = np.stack([np.linalg.norm(D[:,c].reshape(N,-1),axis=1)/np.linalg.norm(Y[:,c].reshape(N,-1),axis=1) for c in range(4)],1)

def ld(d,c,k): return np.load(os.path.join(d,'%s_%s.npy'%(c,k))).astype(np.float64)
def scatter(v, c, N=65):
    ux = np.unique(np.round(c[:,0],12)); uy = np.unique(np.round(c[:,1],12))
    col = np.searchsorted(ux, np.round(c[:,0],12)); row = np.searchsorted(uy, np.round(c[:,1],12))
    img = np.full((N,N), np.nan); img[row,col] = v
    return img[:64,:64]
def deramp(f):
    n = f.shape[0]
    A = np.column_stack([np.ones(n*n), np.repeat(np.linspace(0,1,n), n)])
    cc, *_ = np.linalg.lstsq(A, f.ravel(), rcond=None)
    return (f.ravel() - A@cc).reshape(n,n)
def p_fixed(d):
    pb = scatter(np.load(d+'/p_bouss_pert_raw.npy'), np.load(d+'/coords_p_bouss.npy'))
    pc = scatter(np.load(d+'/p_comp_pert_raw.npy'), np.load(d+'/coords_p_comp.npy'))
    return -deramp(pb), deramp(pc)
e0 = []
for d in sorted(glob.glob(B+'/eps0/dataset/sample_*')):
    m = json.load(open(os.path.join(d,'meta.json'))); e = float(m['dT'])
    x = [ld(d,c,'bouss') for c in ['ux','uy','T','p']]; y = [ld(d,c,'comp') for c in ['ux','uy','T','p']]
    x[3], y[3] = p_fixed(d)
    x[2] = (x[2]-(1-e/2))/e; y[2] = (y[2]-(1-e/2))/e
    e0.append([e, float(m['Ra'])] + [np.linalg.norm(x[i]-y[i])/np.linalg.norm(y[i]) for i in range(4)])
E0 = np.array(e0)
Eall = np.concatenate([eps, E0[:,0]]); Rall = np.vstack([R, E0[:,2:6]])
print('600 组随机壁面 + %d 组 eps→0 核验点（ε=0.02/0.05/0.10，Ra=1e3/1e4/1e5）' % len(E0))
out = {}
print('\n%-7s %-34s %-34s' % ('通道','600 组线性 R=aε+b','600 组二次 R=aε+cε²+b'))
for k,c in enumerate(CH):
    y = R[:,k]
    a1,b1 = np.polyfit(eps,y,1); r1 = y-(a1*eps+b1)
    r2_1 = 1-(r1**2).sum()/((y-y.mean())**2).sum()
    A2 = np.column_stack([eps,eps**2,np.ones_like(eps)]); c2,*_ = np.linalg.lstsq(A2,y,rcond=None)
    r2_2 = 1-((y-A2@c2)**2).sum()/((y-y.mean())**2).sum()
    print('%-7s a=%.4f b=%+.4f R²=%.3f        a=%.4f c=%+.4f b=%+.4f R²=%.3f' %
          (c, a1, b1, r2_1, c2[0], c2[1], c2[2], r2_2))
    out[c] = dict(lin=[float(a1),float(b1),float(r2_1)], quad=[float(v) for v in c2]+[float(r2_2)])

print('\n%-7s %-30s %-30s' % ('通道','仅 600 组幂律 R=Aε^p','含 ε→0 点（627 组）幂律'))
for k,c in enumerate(CH):
    y6 = R[:,k]; ya = Rall[:,k]
    for tag,(e,r) in [('600',(eps,y6)),('627',(Eall,ya))]:
        p,i0,rv,pv,se = stats.linregress(np.log(e),np.log(r))
        if tag=='600': s6 = ('p=%.3f±%.3f A=%.4f R²=%.3f'%(p,se,np.exp(i0),rv**2))
        else: sa = ('p=%.3f±%.3f A=%.4f R²=%.3f'%(p,se,np.exp(i0),rv**2)); out[c]['power']=[float(p),float(se),float(np.exp(i0)),float(rv**2)]
    print('%-7s %-30s %-30s' % (c, s6, sa))

print('\n=== 近零核验：只用 ε≤0.5 的点（600 组 + eps→0 点）线性拟合 ===')
for k,c in enumerate(CH):
    m = Eall<=0.5; y = Rall[m,k]; e = Eall[m]
    a,b = np.polyfit(e,y,1); se_b = np.sqrt(((y-(a*e+b))**2).sum()/(m.sum()-2))
    print('  %-7s n=%3d  a=%.4f  b=%+.5f  (R(0)=0 的偏离 %.2f sigma)' % (c,m.sum(),a,b,abs(b)/se_b))
    out[c]['near_zero'] = dict(n=int(m.sum()), a=float(a), b=float(b), b_sigma=float(abs(b)/se_b))
json.dump(out, open(os.path.expanduser('~/nc_run/p7/eps_law.json'),'w'), indent=1)
print('\nsaved ~/nc_run/p7/eps_law.json')