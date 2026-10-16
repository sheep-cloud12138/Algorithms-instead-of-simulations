import torch, numpy as np, io, sys, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CH=["u_x","u_y","T","p"]
X=torch.load("data/train_legacy/X.pt", map_location="cpu", mmap=True)
Y=torch.load("data/train_legacy/Y.pt", map_location="cpu", mmap=True)
Ra=[]
with open("data/train_legacy/per_sample.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f): Ra.append(float(row["Ra"]))
Ra=np.array(Ra); lrz=(np.log10(Ra)-np.log10(Ra).mean())/np.log10(Ra).std()
D=(Y-X).float()
def spatial_modes(A, k=20):
    """A: (m,n) float64. 返回前 k 个空间右奇异向量 (n,k)"""
    G=(A@A.T)/A.shape[0]
    w,U=np.linalg.eigh(G); U=U[:,::-1]; w=w[::-1].clip(0)
    s=np.sqrt(w[:k])+1e-30
    V=(A.T@U[:,:k])/s[None,:]
    return V
print("=== 转移性检验（Δ 与 Y 都测，作为对照）===")
print("用低Ra(<1e4)样本求空间模态，看能重构多少 高Ra(>1e5) 的能量")
print()
print("channel | 目标 | top1  | top5  | top20 | 随机5维基线")
for i,c in enumerate(CH):
    A=D[:,i].reshape(D.shape[0],-1).numpy().astype(np.float64)
    A0=A-A.mean(0,keepdims=True)
    B=np.stack([np.ones_like(lrz),lrz],1)
    coef,_,_,_=np.linalg.lstsq(B,A0,rcond=None); Rd=A0-B@coef
    Yc=Y[:,i].float().reshape(Y.shape[0],-1).numpy().astype(np.float64); Yc=Yc-Yc.mean(0,keepdims=True)
    for name,M in (("Δ_去Ra",Rd),("Y",Yc)):
        lo=M[Ra<1e4]; hi=M[Ra>1e5]
        Vl=spatial_modes(lo,20)
        H=hi-hi.mean(0); tot=(H**2).sum()
        r=[((H@Vl[:,:k])@Vl[:,:k].T).sum()/tot for k in (1,5,20)]
        rng=np.random.default_rng(0)
        b=[]
        for _ in range(5):
            Q,_=np.linalg.qr(rng.standard_normal((H.shape[1],5))); b.append(((H@Q)@Q.T).sum()/tot)
        print(f"  {c:4s}  | {name:8s} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} |  {np.mean(b):.3f}")
