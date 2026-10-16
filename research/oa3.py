import urllib.request, urllib.parse, json, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def oa(title, n=3):
    url="https://api.openalex.org/works?"+urllib.parse.urlencode({"search":title,"per_page":n,"mailto":"research@example.com"})
    req=urllib.request.Request(url,headers={"User-Agent":"research"})
    for _ in range(3):
        try:
            return json.loads(urllib.request.urlopen(req,timeout=45).read())
        except Exception: time.sleep(3)
    return {"results":[]}
def uninv(idx):
    if not idx: return None
    pos={}
    for w,ps in idx.items():
        for p in ps: pos[p]=w
    return " ".join(pos[k] for k in sorted(pos))
for t in ["The Algebra of Units From Buckingham Pi theorem to Latent-Variable Learning",
          "PPIE-Net Physics-inspired parametric input embedding data-efficient surrogate modeling"]:
    print("="*95); print("Q:", t)
    for w in oa(t).get("results",[]):
        print("-"*80)
        print("YEAR:", w.get("publication_year"), "| CITED:", w.get("cited_by_count"))
        print("TITLE:", w.get("title"))
        print("VENUE:", ((w.get("primary_location") or {}).get("source") or {}).get("display_name"))
        print("DOI:", w.get("doi"))
        a=uninv(w.get("abstract_inverted_index"))
        print("ABSTRACT:", (a[:1600] if a else "(no abstract)"))
    time.sleep(1)
