import torch, numpy as np, io, sys, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CH=["u_x","u_y","T","p"]
X=torch.load("data/train_legacy/X.pt", map_location="cpu", mmap=True)
Y=torch.load("data/train_legacy/Y.pt", map_location="cpu", mmap=True)
Ra=[]
with open("data/train_legacy/per_sample.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f): Ra.append(float(row["Ra"]))
Ra=np.array(Ra); lr=np.log10(Ra); lr=(lr-lr.mean())/lr.std()
D=(Y-X).float()
print("=== 扣掉 Ra 依赖之后，修正项还剩多少结构？ ===")
print("channel | 总方差 | Ra线性可解释 | 残差占比 | 残差谱 top1/top5/top20")
for i,c in enumerate(CH):
    A=D[:,i].reshape(D.shape[0],-1).numpy().astype(np.float64)
    A0=A-A.mean(0,keepdims=True)
    # regress on [1, logRa]
    B=np.stack([np.ones_like(lr),lr],1)
    coef,_,_,_=np.linalg.lstsq(B,A0,rcond=None)
    fit=B@coef
    expl=1-((A0-fit)**2).sum()/(A0**2).sum()
    R=A0-fit
    G=(R@R.T)/R.shape[0]
    ev=np.linalg.eigvalsh(G)[::-1].clip(0)
    cum=np.cumsum(ev)/ev.sum()
    print(f"  {c:4s}  |  {A0.var():.4f}  |    {expl*100:5.1f}%     |  {1-expl:5.1%}  |  {cum[0]:.3f} / {cum[4]:.3f} / {cum[19]:.3f}")
print()
print("=== 残差随 Ra 的结构：分区间看残差的空间图样是否改变 ===")
for i,c in enumerate(["T","u_x"]):
    idx=i if c=="T" else 0
    A=D[:,idx].reshape(D.shape[0],-1).numpy().astype(np.float64)
    A0=A-A.mean(0,keepdims=True)
    B=np.stack([np.ones_like(lr),lr],1)
    coef,_,_,_=np.linalg.lstsq(B,A0,rcond=None); R=A0-B@coef
    bins=[(1e2,1e3),(1e3,1e4),(1e4,1e5),(1e5,1e6),(1e6,3e6)]
    print(f"  -- {c} --")
    for lo,hi in bins:
        m=(Ra>=lo)&(Ra<hi)
        if m.sum()<5: continue
        sub=R[m]
        # 归一化到单位范数后看平均图样的幅度（结构是否一致）
        nrm=np.linalg.norm(sub,axis=1,keepdims=True)+1e-30
        un=(sub/nrm).mean(0)
        print(f"    Ra[{lo:.0e},{hi:.0e}) n={m.sum():4d} | 残差rms={np.sqrt((sub**2).mean()):.4f} | 平均单位图样范数={np.linalg.norm(un):.4f}")
