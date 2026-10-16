import json,glob,os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def load(prefix=""):
    R={}
    for f in glob.glob("results/*.json"):
        d=json.load(open(f))
        if "n" not in d: continue
        key=(d["arm"],d.get("target","Y"))
        if prefix=="": 
            if os.path.basename(f).startswith("ood_") or os.path.basename(f)=="smoke.json": continue
            R.setdefault(key,{})[d["n"]]=d["relL2"]
        else:
            if not os.path.basename(f).startswith(prefix): continue
            R.setdefault(key,{})[d["n"]]=(d["relL2_id"],d["relL2_ood"],d["identity_id"],d["identity_ood"])
    return R
ID=load(); OOD=load("ood_")
Ns=[20,50,100,300,1000]
style=[(("none","Y"),"o-","#7f7f7f","Direct FNO(X) -> Y"),
       (("logra","Y"),"s-","#1f77b4","FNO(X, log Ra) -> Y"),
       (("dim6","Y"),"^-","#d62728","FNO(X, 6 physical quantities) -> Y"),
       (("logra","delta"),"D-","#2ca02c","Backbone + correction:  X + FNO(...)")]
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
for k,m,c,lab in style:
    if k in ID:
        xs=[n for n in Ns if n in ID[k]]; ax[0].loglog(xs,[ID[k][n] for n in xs],m,color=c,label=lab,lw=2,ms=7)
ax[0].axhline(0.1159,ls="--",color="#bbbbbb",lw=1.5)
ax[0].text(1000,0.125,"free baseline  Y=X  (11.6%)",fontsize=8,color="#999999",ha="right")
ax[0].set_title("(a) In-distribution  (Ra 100 - 3e6)",fontsize=11)
for k,m,c,lab in style:
    if k in OOD:
        xs=[n for n in Ns if n in OOD[k]]; ax[1].loglog(xs,[OOD[k][n][1] for n in xs],m,color=c,label=lab,lw=2,ms=7)
ax[1].axhline(0.1108,ls="--",color="#bbbbbb",lw=1.5)
ax[1].text(1150,0.100,"free baseline  Y=X  (11.1%)",fontsize=8,color="#999999",ha="right",va="top")
ax[1].set_title("(b) Ra extrapolation: train Ra<=1e4, test Ra>=3e4 (up to 300x)",fontsize=11)
for a in ax:
    a.set_xlabel("Number of high-fidelity training pairs  N",fontsize=11)
    a.set_ylabel("Test relative L2 error",fontsize=11)
    a.grid(alpha=.3,which="both"); a.legend(fontsize=8.5,loc="lower left"); a.set_xticks(Ns)
    a.set_xticklabels([str(n) for n in Ns]); a.set_xlim(18,1200); a.set_ylim(5e-4,3)
plt.tight_layout(); plt.savefig("results/extrapolation.png",dpi=150)
print("saved results/extrapolation.png")
print()
def row(R,k,pair=False):
    def one(n):
        if k in R and n in R[k]:
            v=R[k][n]
            return "%9.4f"%(v[pair] if isinstance(v,tuple) else v)
        return "        -"
    return "%-40s"%("+".join(k))+ "".join(one(n) for n in Ns)
print("%-40s"%""+"".join("%9s"%("N=%d"%n) for n in Ns))
print("--- in-distribution ---")
for k,_,_,l in style: print(("  "+row(ID,k)).ljust(50), l)
print("--- Ra extrapolation (OOD) ---")
for k,_,_,l in style: print(("  "+row(OOD,k,1)).ljust(50), l)
print()
for k,_,_,l in style:
    if k in OOD:
        print("%-38s OOD/ID ratio:"%l, " ".join("N=%-5d %.2fx"%(n,OOD[k][n][1]/OOD[k][n][0]) for n in Ns if n in OOD[k]))
print()
print("OOD: model error / free-baseline error (>1 means WORSE than doing nothing):")
for k,_,_,l in style:
    if k in OOD:
        print("%-38s"%l, " ".join("N=%-5d %.2fx"%(n,OOD[k][n][1]/OOD[k][n][3]) for n in Ns if n in OOD[k]))
