import urllib.request, json, sys
sys.stdout.reconfigure(encoding="utf-8")
def api(u):
    return json.loads(urllib.request.urlopen(u,timeout=45).read().decode("utf-8","ignore"))
try:
    d=api("https://api.github.com/repos/FEniCS/dolfinx/contents/")
    print([x["name"] for x in d])
except Exception as e: print("ERR",e)
