import urllib.request, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def fetch(url):
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"})
    try:
        r=urllib.request.urlopen(req,timeout=60)
        return r.status, r.read().decode("utf-8","replace")
    except Exception as e:
        return "ERR", f"{type(e).__name__}: {e}"
for u in ["https://r.jina.ai/https://www.sciencedirect.com/science/article/pii/S1568494626016935",
          "https://r.jina.ai/https://doi.org/10.1016/j.asoc.2026.116245"]:
    s,b=fetch(u)
    print("="*90); print(u); print("STATUS:",s)
    print(b[:4000] if s!="ERR" else b)
    if s!="ERR": break
