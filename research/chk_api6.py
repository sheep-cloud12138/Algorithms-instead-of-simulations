import urllib.request, re, sys
sys.stdout.reconfigure(encoding="utf-8")
t=urllib.request.urlopen("https://raw.githubusercontent.com/FEniCS/ufl/main/ufl/__init__.py",timeout=60).read().decode("utf-8","ignore")
print("len",len(t))
for kw in ["VectorElement","MixedElement","TensorElement","FiniteElement","abstractelement"]:
    print(kw, "->", len(re.findall(kw,t)))
print("--- __all__ head ---")
m=re.search(r"__all__\s*=\s*\[", t)
print(t[t.find("__all__"):t.find("__all__")+900] if "__all__" in t else "no __all__")
