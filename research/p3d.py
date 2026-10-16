import torch,numpy as np,csv,io,sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
X=torch.load("data/train_legacy/X.pt",map_location="cpu",mmap=True).float().cuda()
Y=torch.load("data/train_legacy/Y.pt",map_location="cpu",mmap=True).float().cuda()
logRa=np.log10(np.array([float(r["Ra"]) for r in csv.DictReader(open("data/train_legacy/per_sample.csv",encoding="utf-8"))]))
def pcs(M,k=20):
    U,S,V=torch.svd_lowrank(M,q=min(k+5,M.shape[1]-1),niter=4); return (M@V[:,:k]).cpu().numpy()
def cv_r2(F,y,k=5,lam=1e-2,seed=0):
    F=(F-F.mean(0))/(F.std(0)+1e-12)
    idx=np.arange(len(F)); rng=np.random.default_rng(seed); rng.shuffle(idx); folds=np.array_split(idx,k); out=[]
    for j in range(k):
        te=folds[j]; tr=np.concatenate([folds[q] for q in range(k) if q!=j])
        Atr=np.c_[np.ones(len(tr)),F[tr]]; Ate=np.c_[np.ones(len(te)),F[te]]
        w=np.linalg.solve(Atr.T@Atr+lam*np.eye(Atr.shape[1]),Atr.T@y[tr]); p=Ate@w
        out.append(1-((y[te]-p)**2).sum()/((y[te]-y[te].mean())**2).sum())
    return float(np.mean(out)),float(np.min(out))
print("从单通道 X（或修正项 Delta）的 20 维线性投影反推 log10(Ra)，5 折 CV：")
for c,n in enumerate(["u_x","u_y","T","p"]):
    mean,worst=cv_r2(pcs(X[:,c].reshape(len(X),-1)),logRa)
    print("  X 的 %-4s 通道 : R^2 均值 %.6f   最差折 %.6f"%(n,mean,worst))
D=(Y-X)
for c,n in enumerate(["u_x","u_y","T","p"]):
    mean,worst=cv_r2(pcs(D[:,c].reshape(len(D),-1)),logRa)
    print("  Delta 的 %-4s 通道 : R^2 均值 %.6f   最差折 %.6f"%(n,mean,worst))
print()
print("对照：用 X 的同一个 20 维投影去预测别的东西（证明不是'什么都行'）：")
logNu=np.log10(np.abs(X[:,2,:,1]-X[:,2,:,0]).mean(dim=1).cpu().numpy())
print("  预测 log(Nu) 的 R^2 = %.6f"%cv_r2(pcs(X[:,2].reshape(len(X),-1)),logNu)[0])
