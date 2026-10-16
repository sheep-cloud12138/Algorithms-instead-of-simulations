import urllib.request, urllib.parse, xml.etree.ElementTree as ET, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
NS={"a":"http://www.w3.org/2005/Atom"}
ids=["2509.02617","1901.06314","2609.38094","2502.02682","2603.12870","2601.08404","2603.06881","2504.18595","2009.11990","2605.07738","2601.18235","2508.21559"]
url="http://export.arxiv.org/api/query?"+urllib.parse.urlencode({"id_list":",".join(ids),"max_results":len(ids)})
raw=urllib.request.urlopen(url,timeout=60).read()
root=ET.fromstring(raw)
for e in root.findall("a:entry",NS):
    t=" ".join(e.find("a:title",NS).text.split())
    d=e.find("a:published",NS).text[:10]
    s=" ".join(e.find("a:summary",NS).text.split())
    i=e.find("a:id",NS).text.split("/abs/")[-1]
    print("="*100); print(f"[{d}] {t}   ({i})"); print(s)
