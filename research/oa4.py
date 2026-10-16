import urllib.request, urllib.parse, json, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def oa(params):
    url="https://api.openalex.org/works?"+urllib.parse.urlencode(params)
    req=urllib.request.Request(url,headers={"User-Agent":"research"})
    for _ in range(3):
        try: return json.loads(urllib.request.urlopen(req,timeout=45).read())
        except Exception: time.sleep(3)
    return {"results":[]}
def uninv(idx):
    if not idx: return None
    pos={}
    for w,ps in idx.items():
        for p in ps: pos[p]=w
    return " ".join(pos[k] for k in sorted(pos))
for p in [{"filter":"title.search:PPIE-Net","per_page":"5","mailto":"research@example.com"},
          {"filter":"title.search:physics-inspired parametric input embedding","per_page":"5","mailto":"research@example.com"}]:
    print("="*95); print("FILTER:", p["filter"])
    for w in oa(p).get("results",[]):
        print("-"*80)
        print("YEAR:",w.get("publication_year"),"| CITED:",w.get("cited_by_count"),"| DOI:",w.get("doi"))
        print("TITLE:",w.get("title"))
        print("VENUE:",((w.get("primary_location") or {}).get("source") or {}).get("display_name"))
        a=uninv(w.get("abstract_inverted_index"))
        print("ABSTRACT:", (a[:1500] if a else "(no abstract)"))
    time.sleep(1)
