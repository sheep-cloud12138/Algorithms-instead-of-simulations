import torch,numpy as np,csv,io,sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
X=torch.load("data/train_legacy/X.pt",map_location="cpu",mmap=True).float()
Ra=np.array([float(r["Ra"]) for r in csv.DictReader(open("data/train_legacy/per_sample.csv",encoding="utf-8"))])
logRa=np.log10(Ra)
T=X[:,2]
print("=== A. 壁面温度（判断哪边是热壁）===")
print("x=0 壁面 T 均值 %.4f ; x=63 壁面 T 均值 %.4f"%(T[:,:,0].mean(),T[:,:,-1].mean()))
hot=0 if T[:,:,0].mean()>T[:,:,-1].mean() else -1
print("热壁在 x=%s"%("0" if hot==0 else "63"))
dTdx = (T[:,:,1]-T[:,:,0]).mean(dim=1) if hot==0 else (T[:,:,-2]-T[:,:,-1]).mean(dim=1)
Nu = dTdx.abs().numpy()
print()
print("=== B. Nu 是否服从 Ra^0.25 标度（证明 X 里含 Ra 信息）===")
A=np.stack([np.ones_like(logRa),logRa],1)
coef,res,_,_=np.linalg.lstsq(A,np.log(Nu),rcond=None)
pred=A@coef; ss=1-((np.log(Nu)-pred)**2).sum()/((np.log(Nu)-np.log(Nu).mean())**2).sum()
print("拟合 log(Nu) = %.4f + %.4f*log(Ra)   R^2=%.4f"%(coef[0],coef[1],ss))
print("corr(Nu, Ra^0.25) = %.4f"%np.corrcoef(Nu,Ra**0.25)[0,1])
print()
print("=== C. 直接从 X 线性回归 logRa（Ridge, 5折 CV）===")
for c,name in [(0,"u_x"),(1,"u_y"),(2,"T"),(3,"p")]:
    F=X[:,c].reshape(len(X),-1)[:,::64].numpy()   # 每通道抽 64 个点
    F=(F-F.mean(0))/(F.std(0)+1e-9)
    idx=np.arange(len(F)); rng=np.random.default_rng(0); rng.shuffle(idx)
    folds=np.array_split(idx,5); r2=[]
    for k in range(5):
        te=folds[k]; tr=np.concatenate([folds[j] for j in range(5) if j!=k])
        Atr=np.c_[np.ones(len(tr)),F[tr]]; Ate=np.c_[np.ones(len(te)),F[te]]
        lam=1e-3
        w=np.linalg.solve(Atr.T@Atr+lam*np.eye(Atr.shape[1]),Atr.T@logRa[tr])
        p=Ate@w
        r2.append(1-((logRa[te]-p)**2).sum()/((logRa[te]-logRa[te].mean())**2).sum())
    print("  %-4s 仅用该通道 64 个采样点 -> logRa 的 CV R^2 = %.3f"%(name,np.mean(r2)))
F=X.reshape(len(X),-1)[:,::16].numpy(); F=(F-F.mean(0))/(F.std(0)+1e-9)
idx=np.arange(len(F)); rng=np.random.default_rng(0); rng.shuffle(idx); folds=np.array_split(idx,5); r2=[]
for k in range(5):
    te=folds[k]; tr=np.concatenate([folds[j] for j in range(5) if j!=k])
    Atr=np.c_[np.ones(len(tr)),F[tr]]; Ate=np.c_[np.ones(len(te)),F[te]]
    w=np.linalg.solve(Atr.T@Atr+1e-3*np.eye(Atr.shape[1]),Atr.T@logRa[tr])
    p=Ate@w; r2.append(1-((logRa[te]-p)**2).sum()/((logRa[te]-logRa[te].mean())**2).sum())
print("  全 4 通道 1024 个采样点 -> logRa 的 CV R^2 = %.3f"%(np.mean(r2)))