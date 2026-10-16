import urllib.request, re, sys
sys.stdout.reconfigure(encoding="utf-8")
u="https://raw.githubusercontent.com/FEniCS/dolfinx/main/CHANGELOG.md"
try:
    t=urllib.request.urlopen(u,timeout=45).read().decode("utf-8","ignore")
    print("len",len(t))
    for kw in ["ufl_cell","VectorElement","MixedElement","basix.ufl","0.7.0","0.6.0"]:
        ms=[m.start() for m in re.finditer(re.escape(kw),t)]
        print("== %s : %d"%(kw,len(ms)))
        for i in ms[:4]:
            print("   |", " ".join(t[max(0,i-180):i+120].split()))
except Exception as e:
    print("ERR",repr(e))
