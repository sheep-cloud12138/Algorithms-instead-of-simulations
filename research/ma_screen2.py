import urllib.request,json,urllib.parse,time,io,sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
UA={"User-Agent":"Mozilla/5.0 (research)"}
def oa_ta(phrase,extra="",n=25):
    f='title_and_abstract.search:"%s"'%phrase
    if extra: f+=","+extra
    u="https://api.openalex.org/works?filter="+urllib.parse.quote(f,safe=':,')+"&per-page=%d&mailto=r@example.com&select=title,publication_year,doi,primary_location,abstract_inverted_index"%n
    try: d=json.loads(urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60).read().decode("utf-8","replace"))
    except Exception as e: return "ERR:"+str(e)[:100],[]
    out=[]
    for w in d.get("results",[]):
        src=(w.get("primary_location") or {}).get("source") or {}
        ab=w.get("abstract_inverted_index")
        txt=""
        if ab:
            pos={}
            for word,ii in ab.items():
                for i in ii: pos[i]=word
            txt=" ".join(pos[k] for k in sorted(pos))[:400]
        out.append((w.get("publication_year"),w.get("title"),src.get("display_name"),w.get("doi"),txt))
    return d.get("meta",{}).get("count"),out
Q=[("non-Oberbeck-Boussinesq",""),
   ("non-Oberbeck-Boussinesq","title_and_abstract.search:machine learning OR neural OR surrogate"),
   ("low Mach number natural convection",""),
   ("compressibility effects natural convection",""),
   ("Boussinesq approximation machine learning heat transfer",""),
   ("similarity breaking convection",""),
   ("variable density effects turbulent convection neural","")]
for ph,ex in Q:
    c,rows=oa_ta(ph,ex)
    print("="*110); print("QUERY: %s | %s   -> count=%s"%(ph,ex,c))
    for y,t,v,doi,ab in rows[:12]:
        print("   - (%s) %s | %s | %s"%(y,t,v,doi))
        if ab: print("       ABS: %s"%ab[:260])
    time.sleep(0.8)