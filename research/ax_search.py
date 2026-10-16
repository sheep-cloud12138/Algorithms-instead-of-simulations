import urllib.request, urllib.parse, sys, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")
def q(s, n=8):
    u = "http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(s) + "&start=0&max_results=%d&sortBy=relevance" % n
    d = urllib.request.urlopen(u, timeout=45).read()
    r = ET.fromstring(d); ns = {"a": "http://www.w3.org/2005/Atom"}
    print("#### QUERY:", s)
    for e in r.findall("a:entry", ns):
        t = " ".join(e.find("a:title", ns).text.split())
        p = e.find("a:published", ns).text[:7]
        i = e.find("a:id", ns).text
        print(" -", p, "|", t, "|", i)
    print()
qs = ['all:"natural convection" AND all:radiation AND all:surrogate',
      'all:emissivity AND all:"natural convection" AND all:"neural network"',
      'all:"dimensional analysis" AND all:"neural network" AND all:"heat transfer"',
      'all:"multi-fidelity" AND all:convection AND all:"neural operator"',
      'all:"wall radiation" AND all:"natural convection" AND all:"surrogate model"']
for s in qs:
    try:
        q(s)
    except Exception as ex:
        print("ERR", s, ex)
