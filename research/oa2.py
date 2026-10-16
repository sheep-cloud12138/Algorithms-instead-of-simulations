import urllib.request, urllib.parse, json, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def oa(q, n=6):
    url="https://api.openalex.org/works?"+urllib.parse.urlencode({"search":q,"per_page":n,"mailto":"research@example.com"})
    req=urllib.request.Request(url,headers={"User-Agent":"research"})
    for _ in range(3):
        try:
            return json.loads(urllib.request.urlopen(req,timeout=45).read())
        except Exception as e:
            time.sleep(3)
    return {"results":[]}
qs=["dimensionless neural operator surrogate heat transfer",
    "dimensional analysis inductive bias machine learning physics",
    "Buckingham Pi theorem machine learning surrogate model",
    "similarity theory data-efficient surrogate model natural convection",
    "non-dimensional input neural network heat transfer prediction",
    "scale invariance neural operator partial differential equation"]
for q in qs:
    print("="*95); print("Q:", q)
    d=oa(q)
    for w in d.get("results",[]):
        src=((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        print(f"  {w.get('publication_year')} | cited {w.get('cited_by_count')} | {(w.get('title') or '')[:105]} | {src}")
    time.sleep(1)
