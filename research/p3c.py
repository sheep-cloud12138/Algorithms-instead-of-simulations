import torch,numpy as np,csv,io,sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
X=torch.load("data/train_legacy/X.pt",map_location="cpu",mmap=True).float()
logRa=np.log10(np.array([float(r["Ra"]) for r in csv.DictReader(open("data/train_legacy/per_sample.csv",encoding="utf-8"))]))
CH=["u_x","u_y","T","p"]
def cv_r2(F,y,k=5,lam=1e-2,seed=0):
    F=(F-F.mean(0))/(F.std(0)+1e-12)
    idx=np.arange(len(F)); rng=np.random.default_rng(seed); rng.shuffle(idx)
    folds=np.array_split(idx,k); out=[]
    for j in range(k):
        te=folds[j]; tr=np.concatenate([folds[q] for q in range(k) if q!=j])
        Atr=np.c_[np.ones(len(tr)),F[tr]]; Ate=np.c_[np.ones(len(te)),F[te]]
        w=np.linalg.solve(Atr.T@Atr+lam*np.eye(Atr.shape[1]),Atr.T@y[tr])
        p=Ate@w; out.append(1-((y[te]-p)**2).sum()/((y[te]-y[te].mean())**2).sum())
    return float(np.mean(out))
print("=== 从 X 反推 logRa：PCA 前 20 个主成分 → Ridge（5折CV）===")
print("通道 | 用 64 个对角线点 | 用 PCA-20 主成分 | 该通道第1主成分与logRa相关")
for c,name in enumerate(CH):
    M=X[:,c].reshape(len(X),-1).numpy()
    M=M-M.mean(0)
    diag=M[:, ::64]
    G=(M@M.T)/len(M); w,U=np.linalg.eigh(G.astype(np.float64)); U=U[:,::-1][:,:20]
    P=(M@(M.T@U))/len(M)**0.5
    r1=np.corrcoef(P[:,0],logRa)[0,1]
    print("  %-4s |    R^2=%+.3f     |   R^2=%+.3f       |  %+.3f"%(name,cv_r2(diag,logRa),cv_r2(P,logRa),r1))
F=X.reshape(len(X),-1).numpy(); F=F-F.mean(0)
G=(F@F.T)/len(F); w,U=np.linalg.eigh(G.astype(np.float64)); U=U[:,::-1][:,:20]
P=(F@(F.T@U))/len(F)**0.5
print("  %-4s |                  |   R^2=%+.3f"%("全部",cv_r2(P,logRa)))
print()
print("=== 对照：修正在分布内是否也需要参数？用同样的 PCA-Ridge 反推 Y 的不可约部分 ===")
Y=torch.load("data/train_legacy/Y.pt",map_location="cpu",mmap=True).float()
D=(Y-X).reshape(len(X),-1).numpy()
G=(D@D.T)/len(D); w,U=np.linalg.eigh(G.astype(np.float64)); U=U[:,::-1][:,:20]
P=(D@(D.T@U))/len(D)**0.5
print("  修正项 Delta 的第1主成分与 logRa 相关 = %+.3f"%np.corrcoef(P[:,0],logRa)[0,1])
print("  Delta 前20主成分能解释其能量的 %.1f%%"%(100*w[::-1][:20].sum()/w.sum()))
Xf=X.reshape(len(X),-1).numpy(); Gx=(Xf@Xf.T)/len(Xf); wx,Ux=np.linalg.eigh(Gx.astype(np.float64))
print("  X    前20主成分能解释其能量的 %.1f%%"%(100*wx[::-1][:20].sum()/wx.sum()))