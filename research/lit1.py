import urllib.request, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
def get(url):
    req=urllib.request.Request(url,headers=UA)
    try:
        r=urllib.request.urlopen(req,timeout=60); return r.status, r.read().decode("utf-8","replace")
    except Exception as e:
        return "ERR", f"{type(e).__name__}: {e}"
for u in ["https://r.jina.ai/https://arxiv.org/abs/2204.00887",
          "https://r.jina.ai/https://arxiv.org/abs/2202.04643"]:
    s,b=get(u)
    print("="*95); print(u); print("STATUS:",s)
    if s=="ERR": print(b); continue
    print(b[:5000])
