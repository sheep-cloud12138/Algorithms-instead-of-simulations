import subprocess, json, os, sys, time
os.makedirs("results",exist_ok=True)
RA_MAX=1e4; TE_MIN=3e4
cfgs=[]
for arm in ["none","logra","dim6"]:
    for n in [100,300]: cfgs.append((arm,"Y",n))
for n in [100,300]: cfgs.append(("logra","delta",n))
cfgs.append(("none","Y",1000)); cfgs.append(("logra","delta",1000))
log=open("results/sweep_ood.log","a",encoding="utf-8")
for arm,tgt,n in cfgs:
    out=f"results/ood_{arm}_{tgt}_{n}.json"
    if os.path.exists(out): print("skip",out,flush=True); continue
    cmd=[sys.executable,"train_fno.py","--arm",arm,"--n",str(n),"--epochs","150","--target",tgt,
         "--ra_max",str(RA_MAX),"--te_ra_min",str(TE_MIN),"--out",out]
    r=subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8",errors="replace")
    if r.returncode!=0:
        print("FAIL",arm,tgt,n,r.stderr[-500:],flush=True); log.write(f"FAIL {arm} {tgt} {n}\n{r.stderr[-500:]}\n"); log.flush(); continue
    d=json.loads(r.stdout.strip().splitlines()[-1])
    line=(f"{arm:6s} {tgt:5s} N={n:5d} | ID(Ra<1e4) {d['relL2_id']:.4f}  OOD(Ra>3e4) {d['relL2_ood']:.4f}"
          f" | identity ID {d['identity_id']:.4f} OOD {d['identity_ood']:.4f} | t={d['train_s']}s")
    print(line,flush=True); log.write(line+"\n"); log.flush()
print("ALL DONE",flush=True); log.close()