import requests, json, time
H = {"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
def probe(name, url, **kw):
    t0=time.time()
    try:
        r = requests.get(url, headers=H, timeout=25, **kw)
        print(f"{name}: HTTP {r.status_code} | {len(r.content)}B | {time.time()-t0:.1f}s")
        return r
    except Exception as e:
        print(f"{name}: FAIL {type(e).__name__} {e} | {time.time()-t0:.1f}s")
        return None

probe("EuropePMC", "https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"query":"radiation natural convection surrogate","format":"json","pageSize":5})
probe("OpenAlex", "https://api.openalex.org/works", params={"search":"radiation natural convection surrogate","per-page":3})
probe("S2-bulk", "https://api.semanticscholar.org/graph/v1/paper/search/bulk", params={"query":"radiation natural convection surrogate","fields":"title,year,venue,externalIds"})
probe("Crossref", "https://api.crossref.org/works", params={"query.bibliographic":"radiation natural convection surrogate","rows":3})
probe("arXiv", "http://export.arxiv.org/api/query", params={"search_query":"all:radiation natural convection surrogate","max_results":3})
probe("Scholar", "https://scholar.google.com/scholar", params={"q":"radiation natural convection surrogate"})
probe("Scopus-no", "https://api.elsevier.com/content/search/scopus", params={"query":"radiation"})
