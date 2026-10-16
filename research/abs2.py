import urllib.request, urllib.parse, xml.etree.ElementTree as ET, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
NS={"a":"http://www.w3.org/2005/Atom"}
ids=["2605.23391","2311.17068","2103.11177","2208.07482","2511.09578","2606.25259","2602.06241","2604.14566","2412.00568"]
url="http://export.arxiv.org/api/query?"+urllib.parse.urlencode({"id_list":",".join(ids),"max_results":len(ids)})
root=ET.fromstring(urllib.request.urlopen(url,timeout=60).read())
for e in root.findall("a:entry",NS):
    print("="*95)
    print("["+e.find("a:published",NS).text[:10]+"]"," ".join(e.find("a:title",NS).text.split()),"("+e.find("a:id",NS).text.split("/abs/")[-1]+")")
    print(" ".join(e.find("a:summary",NS).text.split()))
