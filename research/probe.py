import urllib.request, urllib.parse, json, time
def try_url(u):
    req = urllib.request.Request(u, headers={"User-Agent":"Mozilla/5.0 (compatible; lit-review; mailto:lit@example.com)"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e:
        return {"ERR": str(e)}
tests = {
 "S2-search": "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode({"query":"surface radiation natural convection surrogate","limit":3,"fields":"title,year,venue,externalIds"}),
 "S2-bulk": "https://api.semanticscholar.org/graph/v1/paper/search/bulk?" + urllib.parse.urlencode({"query":"surface radiation natural convection surrogate","fields":"title,year,venue,externalIds"}),
 "OA": "https://api.openalex.org/works?" + urllib.parse.urlencode({"filter":"title_and_abstract.search:surface radiation natural convection surrogate","per-page":3}),
}
for k,u in tests.items():
    print(k, "->", json.dumps(try_url(u), ensure_ascii=False)[:400]); print()
    time.sleep(3)
