import urllib.request, sys, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")
ns={"a":"http://www.w3.org/2005/Atom"}
def q(search, n=10):
    u="http://export.arxiv.org/api/query?search_query="+search+"&start=0&max_results=%d&sortBy=relevance"%n
    d=urllib.request.urlopen(u,timeout=45).read()
    r=ET.fromstring(d)
    tot=r.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    ents=r.findall("a:entry",ns)
    print("####",search,"| total:",tot.text if tot is not None else "?","| shown:",len(ents))
    for e in ents:
        print("  -",e.find("a:published",ns).text[:7],"|"," ".join(e.find("a:title",ns).text.split()))
    print()
for s in ["all:radiation+AND+all:%22natural+convection%22",
          "all:%22natural+convection%22+AND+all:surrogate+AND+all:radiation",
          "all:dimensionless+AND+all:equivariance+AND+all:neural",
          "abs:%22neural+operator%22+AND+abs:radiation",
          "abs:conjugate+AND+abs:radiation+AND+abs:surrogate"]:
    try: q(s)
    except Exception as ex: print("ERR",s,repr(ex))
