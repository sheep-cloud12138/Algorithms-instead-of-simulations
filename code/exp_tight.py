"""最差 10 组收紧容差复算：同一 seed，稳态 tol 1e-6 -> 1e-8。
看修正量 R 和场本身变了多少。"""
import numpy as np, json, os
ID = [241,99,349,587,293,89,432,511,344,591]
A = os.path.expanduser('~/nc_run/batch/dataset'); B = os.path.expanduser('~/nc_run/tight/dataset')
def toimg(v,c):
    ux=np.unique(np.round(c[:,0],12)); uy=np.unique(np.round(c[:,1],12))
    img=np.full((len(uy),len(ux)),np.nan)
    img[np.searchsorted(uy,np.round(c[:,1],12)),np.searchsorted(ux,np.round(c[:,0],12))]=v
    if len(ux)==129: img=img[::2,::2]
    return img[:64,:64]
def deramp(f):
    n=f.shape[0]; M=np.column_stack([np.ones(n*n),np.repeat(np.linspace(0,1,n),n)])
    cc,*_=np.linalg.lstsq(M,f.ravel(),rcond=None); return (f.ravel()-M@cc).reshape(n,n)
def pair(d,e):
    g=lambda c,k: toimg(np.load(os.path.join(d,'%s_%s.npy'%(c,k))).astype(np.float64), np.load(os.path.join(d,'coords_%s_%s.npy'%(c,k))))
    x=[g('ux','bouss'),g('uy','bouss'),g('T','bouss')]; y=[g('ux','comp'),g('uy','comp'),g('T','comp')]
    pb=toimg(np.load(d+'/p_bouss_pert_raw.npy'),np.load(d+'/coords_p_bouss.npy'))
    pc=toimg(np.load(d+'/p_comp_pert_raw.npy'),np.load(d+'/coords_p_comp.npy'))
    x.append(-deramp(pb)); y.append(deramp(pc))
    x[2]=(x[2]-(1-e/2))/e; y[2]=(y[2]-(1-e/2))/e
    return np.array(x), np.array(y)
rel=lambda a,b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
print('%-6s %-8s %-9s | %-28s | %-28s' % ('id','eps','Ra','原始 R(ux/uy/θ/p)','收紧后 R(ux/uy/θ/p)'))
dR=[]; dY=[]; dB=[]
for i in ID:
    da='%s/sample_%06d'%(A,i); db='%s/sample_%06d'%(B,i)
    if not os.path.exists(db): print('  id=%d 缺失'%i); continue
    ma=json.load(open(da+'/meta.json')); mb=json.load(open(db+'/meta.json'))
    e=float(ma['dT'])
    xa,ya=pair(da,e); xb,yb=pair(db,e)
    Ra_=float(ma['Ra'])
    D=lambda x,y: np.linalg.norm(y-x,axis=(1,2))/np.linalg.norm(y,axis=(1,2))
    ra, rb = D(xa,ya), D(xb,yb)
    print('%-6d %-8.3f %-9.1f | %-28s | %-28s' % (i, e, Ra_, ' '.join('%.4f'%v for v in ra), ' '.join('%.4f'%v for v in rb)))
    dR.append(np.abs(rb-ra)/ra); dY.append([rel(yb[k],ya[k]) for k in range(4)])
    dB.append([rel(xb[k],xa[k]) for k in range(4)])
    print('        rel_change: 原始 %.2e -> 收紧 %.2e' % (ma['march']['comp'].get('achieved_rel_change',float('nan')), mb['march']['comp'].get('achieved_rel_change',float('nan'))))
dR=np.array(dR); dY=np.array(dY); dB=np.array(dB)
print('\n10 组中位：|ΔR|/R = %s' % ' '.join('%.4f'%v for v in np.median(dR,0)))
print('           |ΔY|/|Y| = %s' % ' '.join('%.2e'%v for v in np.median(dY,0)))
print('           |ΔX|/|X| = %s' % ' '.join('%.2e'%v for v in np.median(dB,0)))
print('\n对照：修正量 R 本身在这些样本上是 %s' % ' '.join('%.3f'%v for v in np.median(np.array([D(*pair('%s/sample_%06d'%(B,i), json.load(open('%s/sample_%06d/meta.json'%(B,i)))['dT'])) for i in ID if os.path.exists('%s/sample_%06d'%(B,i))]),0)))