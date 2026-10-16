import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, csv, argparse, json, time, os, sys
sys.path.insert(0,".")
from fno import FNO
p=argparse.ArgumentParser()
p.add_argument("--arm",default="logra"); p.add_argument("--n",type=int,default=100)
p.add_argument("--epochs",type=int,default=300); p.add_argument("--seed",type=int,default=0)
p.add_argument("--width",type=int,default=48); p.add_argument("--batch",type=int,default=32)
p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--target",default="Y"); p.add_argument("--loss",default="target"); p.add_argument("--out",default="")
p.add_argument("--ra_max",type=float,default=0.0); p.add_argument("--te_ra_min",type=float,default=0.0)
a=p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev="cuda"
def load(split):
    X=torch.load(f"data/{split}/X.pt",map_location="cpu",mmap=True).float()
    Y=torch.load(f"data/{split}/Y.pt",map_location="cpu",mmap=True).float()
    Ra=[]
    with open(f"data/{split}/per_sample.csv",encoding="utf-8") as f:
        for r in csv.DictReader(f): Ra.append(float(r["Ra"]))
    return X,Y,np.array(Ra)
Xtr,Ytr,Ratr=load("train_legacy"); Xte,Yte,Rate=load("test_legacy")
def norm(X,Y):
    lo=X.amin(dim=(2,3),keepdim=True); hi=X.amax(dim=(2,3),keepdim=True)
    sc=(hi-lo).clamp_min(1e-8)
    return (X-lo)/sc,(Y-lo)/sc
Xtr_n,Ytr_n=norm(Xtr,Ytr); Xte_n,Yte_n=norm(Xte,Yte)
PR=0.71
def params(arm,Ra,idx):
    if arm=="none": return torch.zeros(len(Ra),0)
    if arm=="logra": return torch.from_numpy(np.log10(Ra)).float().unsqueeze(1)/6.0
    if arm=="dim6":
        rng=np.random.default_rng(1000+idx)          # 每样本固定的随机物理尺度
        g=np.full(len(Ra),9.81)
        L=10**rng.uniform(-2,0,len(Ra))              # 0.01..1 m
        dT=10**rng.uniform(0,2,len(Ra))              # 1..100 K
        al=10**rng.uniform(-7,-5,len(Ra))            # 1e-7..1e-5 m^2/s
        nu=PR*al
        be=Ra*nu*al/(g*dT*L**3)
        M=np.stack([L,dT,g,be,nu,al],1)
        M=np.log10(np.clip(M,1e-30,None))
        M=(M-M.mean(0))/(M.std(0)+1e-12)
        return torch.from_numpy(M).float()
ntr=len(Xtr); g=torch.Generator().manual_seed(a.seed)
pool=np.where(Ratr<=a.ra_max)[0] if a.ra_max>0 else np.arange(ntr)
if len(pool)<a.n: raise SystemExit(f"train pool {len(pool)} < n {a.n}")
sel=torch.from_numpy(pool)[torch.randperm(len(pool),generator=g)[:a.n]]
Ptr=params(a.arm,Ratr[sel],sel.numpy()); Pte=params(a.arm,Rate,np.arange(10000,10000+len(Rate)))
in_ch=4+Ptr.shape[1]
model=FNO(in_ch,4,width=a.width,modes=12,depth=4).to(dev)
opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=0)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
nstep=max(1,a.n//a.batch)
def infer(Xn,P,bs=64):
    outs=[]
    with torch.no_grad():
        for i in range(0,len(Xn),bs):
            x=Xn[i:i+bs].to(dev); pp=P[i:i+bs].to(dev)
            if pp.shape[1]>0: x=torch.cat([x,pp[:,:,None,None].expand(-1,-1,x.shape[2],x.shape[3])],1)
            outs.append(model(x).cpu())
    return torch.cat(outs)
def relL2(pred,tgt):
    ax=tuple(range(1,pred.dim()))
    d=((pred-tgt)**2).sum(dim=ax); n=(tgt**2).sum(dim=ax).clamp_min(1e-12)
    return (torch.sqrt(d/n)).mean().item()
t0=time.time()
for ep in range(a.epochs):
    perm=torch.randperm(a.n,generator=g)
    model.train()
    for s in range(0,a.n-a.batch+1,a.batch):
        j=sel[perm[s:s+a.batch]]
        xf=Xtr_n[j].to(dev); x=xf; y=(Ytr_n[j]-Xtr_n[j]).to(dev) if a.target=="delta" and a.loss=="target" else Ytr_n[j].to(dev); pp=Ptr[perm[s:s+a.batch]].to(dev)
        if pp.shape[1]>0: x=torch.cat([x,pp[:,:,None,None].expand(-1,-1,64,64)],1)
        pred=model(x)
        if a.loss=="recon" and a.target=="delta": pred=pred+xf
        loss=F.mse_loss(pred,y)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
tr=time.time()-t0
pte=infer(Xte_n,Pte)
if a.target=="delta": pte=pte+Xte_n
r_all=relL2(pte,Yte_n)
rc=[relL2(pte[:,i],Yte_n[:,i]) for i in range(4)]
m_ood=torch.from_numpy(Rate>=a.te_ra_min) if a.te_ra_min>0 else torch.zeros(len(Rate),dtype=torch.bool)
r_id = relL2(pte[~m_ood],Yte_n[~m_ood]).__float__() if int((~m_ood).sum()) else None
r_ood= relL2(pte[m_ood],Yte_n[m_ood]).__float__()   if int(m_ood.sum()) else None
r_idn = relL2(Xte_n[~m_ood],Yte_n[~m_ood]).__float__() if int((~m_ood).sum()) else None
r_oodn= relL2(Xte_n[m_ood],Yte_n[m_ood]).__float__()   if int(m_ood.sum()) else None
res=dict(arm=a.arm,target=a.target,loss=a.loss,n=a.n,epochs=a.epochs,seed=a.seed,width=a.width,
         relL2=r_all, relL2_ch=rc, train_s=round(tr,1),
         ra_max=a.ra_max, te_ra_min=a.te_ra_min, n_id=int((~m_ood).sum()), n_ood=int(m_ood.sum()),
         relL2_id=r_id, relL2_ood=r_ood, identity_id=r_idn, identity_ood=r_oodn)
print(json.dumps(res))
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True)
    with open(a.out,"w") as f: json.dump(res,f)
