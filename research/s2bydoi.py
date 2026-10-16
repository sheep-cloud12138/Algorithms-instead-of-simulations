import urllib.request, json, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def s2doi(doi):
    url=f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,abstract,year,venue,citationCount,externalIds,authors"
    req=urllib.request.Request(url,headers={"User-Agent":"research"})
    for _ in range(4):
        try:
            r=urllib.request.urlopen(req,timeout=45)
            return json.loads(r.read())
        except Exception as e:
            print("  retry:",e); time.sleep(8)
    return None
d=s2doi("10.1016/j.asoc.2026.116245")
print(json.dumps(d, ensure_ascii=False, indent=1)[:3000] if d else "FAILED")
