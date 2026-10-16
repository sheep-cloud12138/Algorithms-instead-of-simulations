"""图 1：破缺律 R = |Y-X|/|Y| ~ eps，四个通道，含 eps->0 的独立核验点。"""
import numpy as np, json, os, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
B = os.path.expanduser('~/nc_run')
z = np.load(B+'/p2/pairs600_fixedp.npz', allow_pickle=True)
X = z['X'].astype(np.float64); Y = z['Y'].astype(np.float64); meta = json.loads(str(z['meta']))
N = len(meta); eps = np.array([m['eps'] for m in meta]); lR = np.log10([m['Ra'] for m in meta])
D = Y - X
R = np.stack([np.linalg.norm(D[:,c].reshape(N,-1),axis=1)/np.linalg.norm(Y[:,c].reshape(N,-1),axis=1) for c in range(4)],1)
def toimg(v,c):
    ux=np.unique(np.round(c[:,0],12)); uy=np.unique(np.round(c[:,1],12))
    img=np.full((len(uy),len(ux)),np.nan)
    img[np.searchsorted(uy,np.round(c[:,1],12)),np.searchsorted(ux,np.round(c[:,0],12))]=v
    if len(ux)==129: img=img[::2,::2]
    return img[:64,:64]
def deramp(f):
    n=f.shape[0]; A=np.column_stack([np.ones(n*n),np.repeat(np.linspace(0,1,n),n)])
    cc,*_=np.linalg.lstsq(A,f.ravel(),rcond=None); return (f.ravel()-A@cc).reshape(n,n)
E0=[];R0=[]
for d in sorted(glob.glob(B+'/eps0/dataset/sample_*')):
    m=json.load(open(d+'/meta.json')); e=float(m['dT'])
    g=[np.load(d+'/%s_bouss.npy'%c).astype(np.float64) for c in ['ux','uy','T']]
    cb=[np.load(d+'/coords_%s_bouss.npy'%c) for c in ['ux','uy','T']]
    h=[np.load(d+'/%s_comp.npy'%c).astype(np.float64) for c in ['ux','uy','T']]
    cp=[np.load(d+'/coords_%s_comp.npy'%c) for c in ['ux','uy','T']]
    x=[toimg(g[i],cb[i]) for i in range(3)]; y=[toimg(h[i],cp[i]) for i in range(3)]
    pb=toimg(np.load(d+'/p_bouss_pert_raw.npy'),np.load(d+'/coords_p_bouss.npy'))
    pc=toimg(np.load(d+'/p_comp_pert_raw.npy'),np.load(d+'/coords_p_comp.npy'))
    x.append(-deramp(pb)); y.append(deramp(pc))
    x[2]=(x[2]-(1-e/2))/e; y[2]=(y[2]-(1-e/2))/e
    E0.append(e); R0.append([np.linalg.norm(x[i]-y[i])/np.linalg.norm(y[i]) for i in range(4)])
E0=np.array(E0); R0=np.array(R0)
CH=[r'$u_x$',r'$u_y$',r'$\theta$',r'$p$']
fig,ax=plt.subplots(2,2,figsize=(11,8.6),dpi=170)
for k in range(4):
    a=ax.flat[k]
    sc=a.scatter(eps,R[:,k],c=lR,cmap='viridis',s=11,alpha=.7,linewidths=0)
    Eall=np.concatenate([eps,E0]); Rall=np.concatenate([R[:,k],R0[:,k]])
    sl,ic,rv,pv,se=stats.linregress(np.log(Eall),np.log(Rall))
    xs=np.linspace(0.015,2.0,200)
    a.plot(xs,np.exp(ic)*xs**sl,'k-',lw=1.9,zorder=6,label=r'$R=%.3f\,\varepsilon^{%.3f}$  $R^2$=%.2f'%(np.exp(ic),sl,rv**2))
    a.plot(E0,R0[:,k],'r*',ms=11,zorder=7,mec='k',mew=.5,label=r'$\varepsilon\to0$ check (27)')
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlim(0.015,2.2); a.set_ylim(2e-4,1.2)
    a.set_xlabel(r'$\varepsilon=\Delta T/T_0$',fontsize=12); a.set_ylabel(r'$\|Y-X\|/\|Y\|$   '+CH[k],fontsize=12)
    a.grid(alpha=.25,which='both'); a.legend(fontsize=8,loc='lower right')
    if k==0: cb=fig.colorbar(sc,ax=a,pad=.02); cb.set_label(r'$\log_{10}Ra$',fontsize=10)
fig.suptitle('Boussinesq-compressible discrepancy collapses on $\\varepsilon=\\Delta T/T_0$ over 5 decades (627 samples)',fontsize=12)
fig.tight_layout(rect=[0,0,1,.96])
p=os.path.expanduser('~/nc_run/p7/fig1_law_v2.png'); fig.savefig(p); print('saved',p)
for k in range(4):
    Eall=np.concatenate([eps,E0]); Rall=np.concatenate([R[:,k],R0[:,k]])
    sl,ic,rv,pv,se=stats.linregress(np.log(Eall),np.log(Rall))
    print('%s  p=%.3f±%.3f  A=%.4f  R2=%.3f'%(CH[k],sl,se,np.exp(ic),rv**2))