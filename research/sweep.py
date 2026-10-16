import subprocess, json, os, sys, time
os.makedirs("results",exist_ok=True)
cfgs=[]
for arm in ["none","logra","dim6"]:
    for n in [20,50,100,300,1000]:
        cfgs.append((arm,"Y",n))
for n in [20,50,100,300,1000]:
    cfgs.append(("logra","delta",n))
log=open("results/sweep.log","a",encoding="utf-8")
rows=[]
for arm,tgt,n in cfgs:
    out=f"results/{arm}_{tgt}_{n}.json"
    if os.path.exists(out):
        rows.append(json.load(open(out))); print("skip",out); continue
    cmd=[sys.executable,"train_fno.py","--arm",arm,"--n",str(n),"--epochs","150",
         "--target",tgt,"--out",out]
    t=time.time(); r=subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8",errors="replace")
    if r.returncode!=0:
        print("FAIL",arm,tgt,n,r.stderr[-400:]); log.write(f"FAIL {arm} {tgt} {n}\n{r.stderr[-400:]}\n"); continue
    d=json.loads(r.stdout.strip().splitlines()[-1]); rows.append(d)
    line=f"{arm:6s} {tgt:5s} N={n:5d} relL2={d['relL2']:.4f} ch={[round(x,4) for x in d['relL2_ch']]} t={d['train_s']}s"
    print(line, flush=True); log.write(line+"\n"); log.flush()
print("ALL DONE"); log.close()
