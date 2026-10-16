import re,sys
sys.stdout.reconfigure(encoding="utf-8")
t=open("paper_2606.txt",encoding="utf-8",errors="ignore").read()
for kw in ["159","584","1000 runs","5{,}000","10{,}000","training set","Table 3"]:
    ms=[m.start() for m in re.finditer(re.escape(kw),t)]
    print("== %s : %d"%(kw,len(ms)))
    for i in ms[:3]:
        print("   |"," ".join(t[max(0,i-330):i+330].split()))
