import urllib.request, re, sys
sys.stdout.reconfigure(encoding="utf-8")
t=urllib.request.urlopen("https://raw.githubusercontent.com/FEniCS/ufl/main/ChangeLog.md",timeout=60).read().decode("utf-8","ignore")
print("len",len(t))
for kw in ["VectorElement","MixedElement","FiniteElement","basix","deprecat","removed","2023"]:
    ms=[m.start() for m in re.finditer(kw,t,re.I)]
    print("== %s : %d"%(kw,len(ms)))
    for i in ms[:4]:
        print("   |"," ".join(t[max(0,i-260):i+220].split()))
