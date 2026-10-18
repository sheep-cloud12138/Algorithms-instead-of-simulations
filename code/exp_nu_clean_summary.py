import os,re,glob,json
import numpy as np
DS=os.path.expanduser('~/nc_run/paired/dataset'); M=6
def scatter(vals,cfile,n):
 c=np.load(cfile); ux=np.unique(np.round(c[:,0],12)); uy=np.unique(np.round(c[:,1],12))
 col=np.searchsorted(ux,np.round(c[:,0],12)); row=np.searchsorted(uy,np.round(c[:,1],12))
 img=np.full((n+1,n+1),np.nan); img[row,col]=vals; return img
def nu(d,kind,eps,n=64):
 T=np.load(os.path.join(d,'T_%s.npy'%kind)).astype(np.float64)
 th=scatter((T-(1-eps/2))/eps,os.path.join(d,'coords_T_%s.npy'%kind),n)
 xw=np.arange(M)*(1./(n-1)); w=np.linalg.solve(np.vander(xw,M,increasing=True).T,np.eye(M)[1])
 return float(-np.nanmean(th[:,:M].dot(w)))
tags={}
for d in sorted(glob.glob(os.path.join(DS,'c*_Ra*'))):
 m=re.match(r'(c\d+_Ra\S+?_\d{4})_e([\d.]+)$',os.path.basename(d))
 if m and os.path.isdir(os.path.join(d,'sample_000000')): tags.setdefault(m.group(1),{})[float(m.group(2))]=os.path.join(d,'sample_000000')
rows=[]
for k in sorted(tags):
 for e,d in tags[k].items():
  x=nu(d,'bouss',e); y=nu(d,'comp',e); rows.append((k,e,x,(y-x)/x))
out={}
for e in (0.4,0.8,1.2,1.6,1.9):
 r=[q for q in rows if abs(q[1]-e)<1e-6]
 iso=[q for q in r if re.search(r'_1100$',q[0])]
 big=[q for q in r if abs(q[2])>1]
 a=np.array([q[3] for q in r]); i=np.array([q[3] for q in iso]); b=np.array([q[3] for q in big])
 out[str(e)]=dict(all=[float(a.mean()),float(a.std(ddof=1))],iso=[float(i.mean()),float(i.std(ddof=1))],
                  big=[float(b.mean()),float(b.std(ddof=1))],n_big=len(b),n_iso=len(i))
 print('eps %.1f  all %+.4f sd %.4f | iso %+.4f sd %.4f (n=%d) | |Nu|>1 %+.4f sd %.4f (n=%d)'%(e,a.mean(),a.std(ddof=1),i.mean(),i.std(ddof=1),len(i),b.mean(),b.std(ddof=1),len(b)))
print('\nper Ra isothermal')
for ra in ('1e2','1e3','1e4','1e5','1e6'):
 k=sorted([q for q in rows if '_Ra'+ra+'_1100' in q[0]],key=lambda z:z[1])
 print('Ra',ra,' '.join('%+.4f'%q[3] for q in k))
json.dump(out,open('论文/nu_clean_summary.json','w'),indent=1)
