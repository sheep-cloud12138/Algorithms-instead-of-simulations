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
def spatial_modes(M,k=20):
    m=M.shape[0]; G=(M@M.T)/m
    w,U=np.linalg.eigh(G); U=U[:,::-1]; w=w[::-1].clip(0)
    return (M.T@U[:,:k])/(np.sqrt(m*w[:k])+1e-30)[None,:]
print("=== 空间模态跨 Ra 的转移性（Δ 与 Y 对照）===")
print("低Ra(<1e4)求模态 → 重构高Ra(>1e5)能量")
print()
print("channel | 目标   | top1  | top5  | top20 | 随机5维| Δ去Ra的降维曲线(top1->top20)")
for i,c in enumerate(CH):
    A=D[:,i].reshape(5000,-1).numpy().astype(np.float64)
    A0=A-A.mean(0,keepdims=True)
    B=np.stack([np.ones_like(lrz),lrz],1)
    coef,_,_,_=np.linalg.lstsq(B,A0,rcond=None); Rd=A0-B@coef
    Yc=Y[:,i].float().reshape(5000,-1).numpy().astype(np.float64); Yc=Yc-Yc.mean(0,keepdims=True)
    out={}
    for name,M in (("dRa",Rd),("Y",Yc)):
        lo=M[Ra<1e4]; hi=M[Ra>1e5]
        V=spatial_modes(lo,20); H=hi-hi.mean(0); tot=(H**2).sum()
        r=[]
        for k in (1,5,20):
            P=(H@V[:,:k])@V[:,:k].T; r.append((P**2).sum()/tot)
        rng=np.random.default_rng(0); b=[]
        for _ in range(5):
            Q,_=np.linalg.qr(rng.standard_normal((H.shape[1],5))); P=(H@Q)@Q.T; b.append((P**2).sum()/tot)
        out[name]=(r,np.mean(b))
    (rd,bd)=out["dRa"]; (ry,by)=out["Y"]
    print(f"  {c:4s}  | Δ去Ra | {rd[0]:.3f} | {rd[1]:.3f} | {rd[2]:.3f} |  {bd:.4f} |  {rd[0]:.2f}->{rd[2]:.2f}")
    print(f"  {c:4s}  | Y     | {ry[0]:.3f} | {ry[1]:.3f} | {ry[2]:.3f} |  {by:.4f} |  {ry[0]:.2f}->{ry[2]:.2f}")
