import urllib.request, re, json, sys
sys.stdout.reconfigure(encoding="utf-8")
def get(branch):
    u="https://raw.githubusercontent.com/FEniCS/ufl/%s/ufl/__init__.py"%branch
    try:
        t=urllib.request.urlopen(u,timeout=45).read().decode("utf-8","ignore")
    except Exception as e:
        return "ERR %r"%e
    return "VectorElement=%d MixedElement=%d"%(len(re.findall("VectorElement",t)),len(re.findall("MixedElement",t)))
tags=json.loads(urllib.request.urlopen("https://api.github.com/repos/FEniCS/ufl/tags?per_page=25",timeout=45).read())
names=[t["name"] for t in tags]
print("UFL tags:",names[:25])
for b in ["2022.2.0","2023.1.0","2023.2.0","2024.1.0","2024.2.0","main"]:
    if b in names or b=="main":
        print("  ",b,"->",get(b))
