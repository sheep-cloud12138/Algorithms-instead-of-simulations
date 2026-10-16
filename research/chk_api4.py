import urllib.request, json, re, sys
sys.stdout.reconfigure(encoding="utf-8")
def txt(u):
    return urllib.request.urlopen(u,timeout=60).read().decode("utf-8","ignore")
def api(u):
    return json.loads(txt(u))
print("--- ufl repo root ---")
try: print([x["name"] for x in api("https://api.github.com/repos/FEniCS/ufl/contents/")])
except Exception as e: print("ERR",e)
print("--- dolfinx releases (tags/names) ---")
try:
    rs=api("https://api.github.com/repos/FEniCS/dolfinx/releases?per_page=12")
    for r in rs:
        print("  ",r["tag_name"],"|",r["name"],"|",r["published_at"][:10])
except Exception as e: print("ERR",e)
