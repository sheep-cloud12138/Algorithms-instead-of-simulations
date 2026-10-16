import urllib.request, json, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
def get(url):
    req=urllib.request.Request(url,headers=UA)
    try:
        r=urllib.request.urlopen(req,timeout=45); return r.status, r.read().decode("utf-8","replace")
    except Exception as e:
        return "ERR", f"{type(e).__name__}: {e}"
s,b=get("https://api.github.com/repos/Nurshat317/NeuralConvection")
print("GITHUB REPO:", s)
if s!="ERR":
    d=json.loads(b)
    print("  full_name:", d.get("full_name"), "| stars:", d.get("stargazers_count"), "| license:", (d.get("license") or {}).get("spdx_id"))
    print("  pushed_at:", d.get("pushed_at"), "| desc:", (d.get("description") or "")[:150])
s2,b2=get("https://huggingface.co/api/datasets?search=NeuralConvection")
print()
print("HF DATASETS:", s2)
if s2!="ERR":
    try:
        for d in json.loads(b2):
            print("  -", d.get("id"), "| downloads:", d.get("downloads"), "| likes:", d.get("likes"))
    except Exception as e: print("  parse:", e, b2[:300])
