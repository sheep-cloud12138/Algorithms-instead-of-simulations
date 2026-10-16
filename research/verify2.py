import torch, numpy as np, csv, sys
sys.stdout.reconfigure(encoding="utf-8")
def load(split):
    X=torch.load(f"data/{split}/X.pt",map_location="cpu",mmap=True).float()
    Y=torch.load(f"data/{split}/Y.pt",map_location="cpu",mmap=True).float()
    Ra=[]
    with open(f"data/{split}/per_sample.csv",encoding="utf-8") as f:
        for r in csv.DictReader(f): Ra.append(float(r["Ra"]))
    return X,Y,np.array(Ra)
X,Y,Ra=load("test_legacy")
lo=X.amin(dim=(2,3),keepdim=True); hi=X.amax(dim=(2,3),keepdim=True); sc=(hi-lo).clamp_min(1e-8)
Xn=(X-lo)/sc; Yn=(Y-lo)/sc
def relL2(p,t):
    ax=tuple(range(1,p.dim()))
    return (torch.sqrt(((p-t)**2).sum(ax)/ (t**2).sum(ax).clamp_min(1e-12))).mean().item()
print("test N =",len(X),"Ra range",Ra.min(),Ra.max())
print("IDENTITY  Y=X  relL2 (all) =", round(relL2(Xn,Yn),4))
m=Ra>=3e4
print("  ID  (Ra<3e4, n=%d) ="%int((~m).sum()), round(relL2(Xn[~m],Yn[~m]),4))
print("  OOD (Ra>=3e4,n=%d) ="%int(m.sum()), round(relL2(Xn[m],Yn[m]),4))
ch=["ux","uy","T","p"]
print("per-channel identity:", {c: round(relL2(Xn[:,i],Yn[:,i]),4) for i,c in enumerate(ch)})
print("delta=Y-X relL2 per channel:", {c: round(relL2(Xn[:,i],Yn[:,i]) if False else torch.sqrt(((Yn[:,i]-Xn[:,i])**2).sum((1,2))/(Yn[:,i]**2).sum((1,2)).clamp_min(1e-12)).median().item(),4) for i,c in enumerate(ch)})
