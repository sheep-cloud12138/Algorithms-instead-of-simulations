import torch, numpy as np, io, sys, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
X=torch.load("data/train_legacy/X.pt", map_location="cpu", mmap=True)
Y=torch.load("data/train_legacy/Y.pt", map_location="cpu", mmap=True)
Ra=[]
with open("data/train_legacy/per_sample.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f): Ra.append(float(row["Ra"]))
Ra=np.array(Ra)
i=2  # T
A=(Y[:,i].float()-X[:,i].float()).reshape(5000,-1).numpy().astype(np.float64)
print("A", A.shape, "rms", np.sqrt((A**2).mean()))
lo=A[Ra<1e4]; hi=A[Ra>1e5]
print("lo",lo.shape,"hi",hi.shape)
G=(lo@lo.T)/lo.shape[0]
print("G shape", G.shape)
w,U=np.linalg.eigh(G); U=U[:,::-1]; w=w[::-1].clip(0)
print("top eigvals:", w[:5], "sum:", w.sum())
s=np.sqrt(w[:20])+1e-30
V=(lo.T@U[:,:20])/s[None,:]
print("V", V.shape, "col norms", np.linalg.norm(V,axis=0)[:5])
H=hi-hi.mean(0)
print("H norm^2", (H**2).sum())
for k in (1,5,20):
    Vk=V[:,:k]
    proj=(H@Vk)@Vk.T
    print(f"k={k} captured {(proj**2).sum()/ (H**2).sum():.6f}")
