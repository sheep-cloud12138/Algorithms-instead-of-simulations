import torch, numpy as np, io, sys, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CH=["u_x","u_y","T","p"]
X=torch.load("data/train_legacy/X.pt", map_location="cpu", mmap=True)
Y=torch.load("data/train_legacy/Y.pt", map_location="cpu", mmap=True)
Ra=[]
with open("data/train_legacy/per_sample.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f): Ra.append(float(row["Ra"]))
Ra=np.array(Ra); lr=np.log10(Ra); lrz=(lr-lr.mean())/lr.std()
D=(Y-X).float()
def modes_for(A, thr=0.99):
    A0=A-A.mean(0,keepdims=True)
    G=(A0@A0.T)/A0.shape[0]
    ev=np.linalg.eigvalsh(G)[::-1].clip(0); cum=np.cumsum(ev)/ev.sum()
    return int(np.searchsorted(cum,thr)+1), cum
print("=== 关键对照：Δ 的秩 vs Y 本身的秩（平滑场天然低秩，必须比）===")
print("channel | 达99%能量所需模态数:  Y_raw | Δ_raw | Δ_去Ra线性 ")
for i,c in enumerate(CH):
    A=D[:,i].reshape(D.shape[0],-1).numpy().astype(np.float64)
    Yc=Y[:,i].float().reshape(Y.shape[0],-1).numpy().astype(np.float64)
    kY,_=modes_for(Yc)
    kD,cumD=modes_for(A)
    A0=A-A.mean(0,keepdims=True)
    B=np.stack([np.ones_like(lrz),lrz],1)
    coef,_,_,_=np.linalg.lstsq(B,A0,rcond=None); R=A0-B@coef
    kR,_=modes_for(R)
    print(f"  {c:4s}  |            {kY:3d}        |  {kD:3d}  |    {kR:3d}")
print()
print("=== P2 决定性检验：低 Ra 上学到的空间模态，能不能重构高 Ra 的修正项？===")
print("channel | 用低Ra模态重构高Ra的捕获能量(top1/top5/top20) | 随机子空间基线")
for i,c in enumerate(CH):
    A=D[:,i].reshape(D.shape[0],-1).numpy().astype(np.float64)
    A0=A-A.mean(0,keepdims=True)
    B=np.stack([np.ones_like(lrz),lrz],1)
    coef,_,_,_=np.linalg.lstsq(B,A0,rcond=None); R=A0-B@coef
    lo=R[Ra<1e4]; hi=R[Ra>1e5]
    if len(hi)<20: continue
    Gl=(lo-lo.mean(0))@(lo-lo.mean(0)).T/len(lo)
    w,V=np.linalg.eigh(Gl); V=V[:,::-1]
    H=(hi-hi.mean(0))
    tot=(H**2).sum()
    res=[]
    for k in (1,5,20):
        Vk=V[:,:k]; res.append(((H@Vk)@Vk.T).sum()/tot)
    rng=np.random.default_rng(0); base=[]
    for _ in range(5):
        Q,_=np.linalg.qr(rng.standard_normal((H.shape[1],5)))
        base.append(((H@Q)@Q.T).sum()/tot)
    print(f"  {c:4s}  |  {res[0]:.3f} / {res[1]:.3f} / {res[2]:.3f}   |  {np.mean(base):.3f} (随机5维)")
