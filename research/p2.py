import torch, numpy as np, io, sys, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CH=["u_x","u_y","T","p"]
X=torch.load("data/train_legacy/X.pt", map_location="cpu", mmap=True)
Y=torch.load("data/train_legacy/Y.pt", map_location="cpu", mmap=True)
Ra=[]; rel={c:[] for c in CH}
with open("data/train_legacy/per_sample.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        Ra.append(float(row["Ra"]))
        for c,k in zip(CH,["relL2_u_x","relL2_u_y","relL2_T","relL2_p"]):
            rel[c].append(float(row[k]))
Ra=np.array(Ra)
print("N =", X.shape[0], "| Ra range: %.3g .. %.3g (median %.3g)"%(Ra.min(),Ra.max(),np.median(Ra)))
print()
print("=== 1) 修正项幅度 vs Ra ===")
print("channel | median relL2 | corr(log10 Ra, relL2) | relL2@Ra<1e3 | relL2@Ra>1e6")
for c in CH:
    v=np.array(rel[c]); lo=v[Ra<1e3]; hi=v[Ra>1e6]
    cc=np.corrcoef(np.log10(Ra), v)[0,1]
    print(f"  {c:4s}  |   {np.median(v):.4e}   |        {cc:+.3f}        |  {np.median(lo):.3e}  |  {np.median(hi):.3e}")
print()
D=(Y-X).float()
print("=== 2) 修正项的空间秩结构（奇异值谱）===")
print("channel | top1 | top5 | top20 | top100  (累计能量占比)")
for i,c in enumerate(CH):
    A=D[:,i].reshape(D.shape[0],-1)
    G=(A@A.T)/A.shape[0]
    ev=torch.linalg.eigvalsh(G).flip(0).clamp(min=0)
    tot=ev.sum()
    cum=torch.cumsum(ev,0)/tot
    print(f"  {c:4s}  | {cum[0]:.3f} | {cum[4]:.3f} | {cum[19]:.3f} | {cum[99]:.3f}")
print()
print("=== 3) 修正项的空间集中度 vs 场本身 ===")
am=D.abs().mean(0)   # (4,64,64)
ym=Y.abs().float().mean(0)
for i,c in enumerate(CH):
    a=am[i]; 
    tot=a.sum()
    # 边界两列 + 边界两行 占比
    edge=(a[:,0].sum()+a[:,-1].sum()+a[0,:].sum()+a[-1,:].sum())/tot
    print(f"  {c:4s}  | 边界4圈占|d|总量: {edge*100:5.1f}%   |  d 的相对幅度 rms: {D[:,i].std().item()/Y[:,i].std().item():.4f}")
