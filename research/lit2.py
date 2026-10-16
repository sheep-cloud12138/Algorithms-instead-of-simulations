import urllib.request, urllib.parse, xml.etree.ElementTree as ET, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
NS={"a":"http://www.w3.org/2005/Atom"}
ids=["2204.00887","2202.04643","2106.06610","2312.10100"]
url="http://export.arxiv.org/api/query?"+urllib.parse.urlencode({"id_list":",".join(ids),"max_results":len(ids)})
for _ in range(4):
    try:
        raw=urllib.request.urlopen(url,timeout=60).read(); break
    except Exception as e:
        print("retry",e,file=sys.stderr); time.sleep(5)
root=ET.fromstring(raw)
for e in root.findall("a:entry",NS):
    print("="*95)
    print("["+e.find("a:published",NS).text[:10]+"]"," ".join(e.find("a:title",NS).text.split()),"("+e.find("a:id",NS).text.split("/abs/")[-1]+")")
    print(" ".join(e.find("a:summary",NS).text.split()))
