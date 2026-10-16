import urllib.request, sys
sys.stdout.reconfigure(encoding="utf-8")
u="http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=2"
try:
    d=urllib.request.urlopen(u,timeout=45).read()
    print("LEN",len(d)); print(d[:600].decode("utf-8","ignore"))
except Exception as ex:
    print("ERR",repr(ex))
