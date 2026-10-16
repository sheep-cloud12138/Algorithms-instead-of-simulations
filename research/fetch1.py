import urllib.request, urllib.parse, json, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
def fetch(url, n=6000):
    req=urllib.request.Request(url,headers=UA)
    try:
        r=urllib.request.urlopen(req,timeout=45)
        return r.status, r.read().decode("utf-8","replace")[:n]
    except Exception as e:
        return "ERR", f"{type(e).__name__}: {e}"
for u in ["https://api.unpaywall.org/v2/10.1016/j.asoc.2026.116245?email=research@example.com",
          "https://www.sciencedirect.com/science/article/pii/S1568494626016935"]:
    s,b=fetch(u)
    print("="*90); print(u); print("STATUS:", s)
    if s=="ERR": print(b); continue
    if "unpaywall" in u:
        try:
            d=json.loads(b if isinstance(b,str) else "")
        except Exception:
            d=None
    print(b[:2500])
