import urllib.request, urllib.parse, json, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def get(url):
    req=urllib.request.Request(url,headers={"User-Agent":"research"})
    for _ in range(3):
        try: return urllib.request.urlopen(req,timeout=45).read().decode("utf-8","replace")
        except Exception as e: time.sleep(3)
    return ""
def uninv(idx):
    if not idx: return None
    pos={}
    for w,ps in idx.items():
        for p in ps: pos[p]=w
    return " ".join(pos[k] for k in sorted(pos))
u="https://api.openalex.org/works?"+urllib.parse.urlencode({"search":"PPIE-Net physics-inspired parametric input embedding surrogate","per_page":"6","mailto":"research@example.com"})
d=json.loads(get(u))
for w in d.get("results",[]):
    print("-"*80)
    print("YEAR:",w.get("publication_year"),"| CITED:",w.get("cited_by_count"),"| DOI:",w.get("doi"))
    print("TITLE:",w.get("title"))
    print("VENUE:",((w.get("primary_location") or {}).get("source") or {}).get("display_name"))
    a=uninv(w.get("abstract_inverted_index"))
    print("ABSTRACT:", (a[:1200] if a else "(no abstract)"))
