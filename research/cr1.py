import urllib.request, urllib.parse, json, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def cr(q, rows=5):
    url="https://api.crossref.org/works?"+urllib.parse.urlencode({"query.bibliographic":q,"rows":rows,"mailto":"research@example.com"})
    req=urllib.request.Request(url,headers={"User-Agent":"research/1.0 (mailto:research@example.com)"})
    for _ in range(3):
        try:
            return json.loads(urllib.request.urlopen(req,timeout=45).read())
        except Exception as e:
            print("  retry:",e); time.sleep(4)
    return {"message":{"items":[]}}
for q in ["PPIE-Net Physics-inspired parametric input embedding data-efficient surrogate modeling",
          "physics-inspired parametric input embedding data-efficient surrogate modeling parametric"]:
    print("="*95); print("Q:",q)
    d=cr(q)
    for it in d["message"]["items"]:
        print("-"*70)
        print("YEAR:", (it.get("issued",{}).get("date-parts") or [[None]])[0][0], "| DOI:", it.get("DOI"))
        print("TITLE:", (it.get("title") or [""])[0])
        print("VENUE:", (it.get("container-title") or [""])[0][:90])
        ab=it.get("abstract")
        if ab: print("ABSTRACT:", ab[:1200])
