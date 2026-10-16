import urllib.request, urllib.parse, sys, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")
def q(s, n=10):
    u = "http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(s) + "&start=0&max_results=%d&sortBy=relevance" % n
    d = urllib.request.urlopen(u, timeout=45).read()
    r = ET.fromstring(d); ns = {"a": "http://www.w3.org/2005/Atom"}
    ents = r.findall("a:entry", ns)
    print("#### QUERY:", s, "->", len(ents))
    for e in ents:
        t = " ".join(e.find("a:title", ns).text.split())
        p = e.find("a:published", ns).text[:7]
        print("  -", p, "|", t)
    print()
qs = ["all:rayleigh-benard+AND+all:radiation+AND+all:surrogate",
      "abs:natural+convection+AND+abs:radiation+AND+abs:neural",
      "abs:dimensionless+AND+abs:equivariance",
      "abs:physics-informed+AND+abs:convection+AND+abs:radiation",
      "abs:multifidelity+AND+abs:heat+AND+abs:surrogate"]
for s in qs:
    try:
        q(s)
    except Exception as ex:
        print("ERR", s, ex)
