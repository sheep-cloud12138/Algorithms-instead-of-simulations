import urllib.request, urllib.parse, json, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
url="https://api.openalex.org/works?"+urllib.parse.urlencode({"search":"PPIE-Net physics-inspired parametric input embedding","per_page":"5","mailto":"research@example.com"})
print("URL:", url)
req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 research"})
try:
    r=urllib.request.urlopen(req,timeout=45)
    body=r.read().decode("utf-8","replace")
    print("STATUS:", r.status, "LEN:", len(body))
    d=json.loads(body)
    for w in d.get("results",[]):
        print("-"*70)
        print(w.get("publication_year"), "|", w.get("doi"))
        print(w.get("title"))
        print(((w.get("primary_location") or {}).get("source") or {}).get("display_name"))
except Exception as e:
    print("ERR:", type(e).__name__, e)
