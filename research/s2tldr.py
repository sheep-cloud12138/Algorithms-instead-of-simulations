import urllib.request, urllib.parse, json, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def s2(url):
    req=urllib.request.Request(url,headers={"User-Agent":"research"})
    for _ in range(4):
        try: return json.loads(urllib.request.urlopen(req,timeout=45).read())
        except Exception as e:
            print("  retry:",e); time.sleep(8)
    return None
d=s2("https://api.semanticscholar.org/graph/v1/paper/9f2cfd4c842791ddd3947d14f9afa4066235f906?fields=title,tldr,abstract,fieldsOfStudy,publicationTypes")
print("TLDR/ABSTRACT:")
print(json.dumps(d, ensure_ascii=False, indent=1)[:2500])
print()
d2=s2("https://api.semanticscholar.org/graph/v1/paper/search?"+urllib.parse.urlencode({"query":"PPIE-Net physics-inspired parametric input embedding","limit":5,"fields":"title,year,venue,tldr,abstract"}))
print("SEARCH:")
print(json.dumps(d2, ensure_ascii=False, indent=1)[:2500])
