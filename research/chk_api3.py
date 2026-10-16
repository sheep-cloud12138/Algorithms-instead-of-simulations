import urllib.request, re, sys
sys.stdout.reconfigure(encoding="utf-8")
u="https://raw.githubusercontent.com/FEniCS/dolfinx/main/ChangeLog.rst"
t=urllib.request.urlopen(u,timeout=60).read().decode("utf-8","ignore")
print("len",len(t))
for kw in ["ufl_cell","VectorElement","MixedElement","basix.ufl","mixed_element"]:
    ms=[m.start() for m in re.finditer(re.escape(kw),t)]
    print("== %s : %d"%(kw,len(ms)))
    for i in ms[:5]:
        print("   |", " ".join(t[max(0,i-260):i+160].split()))
