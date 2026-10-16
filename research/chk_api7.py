import urllib.request, re, sys
sys.stdout.reconfigure(encoding="utf-8")
t=urllib.request.urlopen("https://raw.githubusercontent.com/FEniCS/dolfinx/main/python/dolfinx/mesh.py",timeout=60).read().decode("utf-8","ignore")
print("dolfinx main mesh.py len",len(t))
for kw in ["def ufl_cell","ufl_cell","basix_cell","def create_rectangle"]:
    print(" ",kw,"->",len(re.findall(re.escape(kw),t)))
