import urllib.request, urllib.parse, json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def oa(q, n=5):
    url="https://api.openalex.org/works?"+urllib.parse.urlencode({"search":q,"per_page":n,"mailto":"research@example.com"})
    req=urllib.request.Request(url,headers={"User-Agent":"research"})
    try:
        d=json.loads(urllib.request.urlopen(req,timeout=45).read())
    except Exception as e:
        return "ERR "+str(e)
    out=[]
    for w in d.get("results",[]):
        out.append((w.get("publication_year"), w.get("cited_by_count"), (w.get("title") or "")[:110], (w.get("primary_location") or {}).get("source",{}) and ((w.get("primary_location") or {}).get("source") or {}).get("display_name")))
    return out
for q in ["Natural convection of air in a square cavity a bench mark numerical solution",
          "Boussinesq natural convection square cavity benchmark neural network surrogate"]:
    print("###", q)
    r=oa(q)
    if isinstance(r,str): print("  ",r)
    else:
        for y,c,t,s in r: print(f"  {y} | cited {c} | {t} | {s}")
