import urllib.request, sys
sys.stdout.reconfigure(encoding="utf-8")
t=urllib.request.urlopen("https://raw.githubusercontent.com/FEniCS/dolfinx/main/python/dolfinx/mesh.py",timeout=60).read().decode("utf-8","ignore").splitlines()
for i,l in enumerate(t):
    if "ufl_cell" in l or "basix_cell" in l:
        print("\n".join("%4d| %s"%(j+1,t[j]) for j in range(max(0,i-6),min(len(t),i+8))))
        print("-----")
