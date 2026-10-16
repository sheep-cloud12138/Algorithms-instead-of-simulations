import urllib.request, urllib.parse, xml.etree.ElementTree as ET, sys, time
NS={"a":"http://www.w3.org/2005/Atom"}
ids = sys.argv[1:]
url="http://export.arxiv.org/api/query?"+urllib.parse.urlencode({"id_list":",".join(ids),"max_results":len(ids)})
raw=None
for _ in range(4):
    try: raw=urllib.request.urlopen(url,timeout=45).read(); break
    except Exception as e: print("retry",e); time.sleep(5)
root=ET.fromstring(raw)
for e in root.findall("a:entry",NS):
    print("="*100)
    print("ID:", e.find("a:id",NS).text.strip())
    print("DATE:", e.find("a:published",NS).text[:10])
    print("TITLE:", " ".join(e.find("a:title",NS).text.split()))
    print("AUTHORS:", ", ".join(a.find("a:name",NS).text for a in e.findall("a:author",NS))[:300])
    jr = e.find("{http://arxiv.org/schemas/atom}journal_ref")
    doi = e.find("{http://arxiv.org/schemas/atom}doi")
    print("JOURNAL_REF:", jr.text if jr is not None else None, "| DOI:", doi.text if doi is not None else None)
    print("ABS:", " ".join(e.find("a:summary",NS).text.split()))
