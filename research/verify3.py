import torch, csv, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
X=torch.load("data/train_legacy/X.pt",map_location="cpu",mmap=True).float()
Y=torch.load("data/train_legacy/Y.pt",map_location="cpu",mmap=True).float()
ch=["ux","uy","T","p"]
def med_relL2(p,t):
    return (torch.sqrt(((p-t)**2).sum((1,2))/(t**2).sum((1,2)).clamp_min(1e-30))).median().item()
print("RAW space  ||Y-X||/||Y|| median:", {c: round(med_relL2(Y[:,i],X[:,i]) if False else med_relL2(X[:,i],Y[:,i]),4) for i,c in enumerate(ch)})
print("RAW space  ||Y-X||/||X|| median:", {c: round(med_relL2(Y[:,i],X[:,i]),4) for i,c in enumerate(ch)})
# magnitude of fields
for i,c in enumerate(ch):
    print(c,"| |X| rms", round(float(X[:,i].pow(2).mean().sqrt()),4), "| |Y| rms", round(float(Y[:,i].pow(2).mean().sqrt()),4), "| |Y-X| rms", round(float((Y[:,i]-X[:,i]).pow(2).mean().sqrt()),4))
