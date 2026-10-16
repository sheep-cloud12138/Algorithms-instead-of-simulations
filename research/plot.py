import json,glob,io,sys,os
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R={}
for f in glob.glob("results/*.json"):
    if os.path.basename(f)=="smoke.json": continue
    d=json.load(open(f))
    if "n" not in d or "relL2" not in d: continue
    if d.get("loss","target")!="target": continue
    R.setdefault((d["arm"],d.get("target","Y")),{})[d["n"]]=d["relL2"]
Ns=[20,50,100,300,1000]
style=[(("none","Y"),"o-","#7f7f7f","No parameter:  FNO(X) -> Y"),
       (("logra","Y"),"s-","#1f77b4","Dimensionless:  FNO(X, log Ra) -> Y"),
       (("dim6","Y"),"^-","#d62728","Dimensional:  FNO(X, 6 raw quantities) -> Y"),
       (("logra","delta"),"D-","#2ca02c","Physics backbone + correction:  X + FNO(...)")]
fig,ax=plt.subplots(figsize=(7.6,5.2))
for k,m,c,lab in style:
    if k in R:
        xs=[n for n in Ns if n in R[k]]; ys=[R[k][n] for n in xs]
        ax.loglog(xs,ys,m,color=c,label=lab,lw=2,ms=7)
ax.axhline(0.1159,ls="--",color="#bbbbbb",lw=1.5)
ax.text(1150,0.125,"identity baseline  Y=X  (11.6%)",fontsize=8,color="#999999",ha="right")
ax.set_xlabel("Number of training samples  N",fontsize=12)
ax.set_ylabel("Test relative L2 error",fontsize=12)
ax.set_title("Natural convection in a square cavity:  Boussinesq -> compressible\nFNO, 64x64, single RTX 4060 Laptop",fontsize=11)
ax.grid(alpha=.3,which="both"); ax.legend(fontsize=9,loc="lower left")
ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns]); ax.set_xlim(18,1200); ax.set_ylim(5e-4,3)
plt.tight_layout(); plt.savefig("results/validation.png",dpi=150)
print("saved results/validation.png"); print()
print("%-44s"%""+"".join("%9s"%f"N={n}" for n in Ns))
for k,m,c,lab in style:
    if k in R: print("%-44s"%lab[:42]+"".join("%9.4f"%R[k].get(n,float('nan')) for n in Ns))
print("%-44s"%("identity  Y=X")+"%9.4f"%0.1159); print()
print("correction arm speedup vs dimensionless-direct:")
for n in Ns:
    a=R.get(("logra","Y"),{}).get(n); b=R.get(("logra","delta"),{}).get(n)
    if a and b: print("  N=%5d : %.2fx"%(n,a/b))
print("dimensional vs dimensionless error ratio:")
for n in Ns:
    a=R.get(("logra","Y"),{}).get(n); b=R.get(("dim6","Y"),{}).get(n)
    if a and b: print("  N=%5d : %.2fx"%(n,b/a))
print("no-param vs dimensionless ratio:")
for n in Ns:
    a=R.get(("logra","Y"),{}).get(n); b=R.get(("none","Y"),{}).get(n)
    if a and b: print("  N=%5d : %.2fx"%(n,b/a))
