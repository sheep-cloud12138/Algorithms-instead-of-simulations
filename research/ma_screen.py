import urllib.request,json,urllib.parse,time,io,sys,os
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
os.makedirs("lit",exist_ok=True)
UA={"User-Agent":"Mozilla/5.0 (research screening)"}
def get(u,t=45):
    try:
        return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=t).read().decode("utf-8","replace")
    except Exception as e:
        return "ERR:"+type(e).__name__+":"+str(e)[:120]
def s2(q,n=15):
    u="https://api.semanticscholar.org/graph/v1/paper/search/bulk?query="+urllib.parse.quote(q)+"&fields=title,year,venue,externalIds,abstract&limit=100"
    r=get(u)
    if r.startswith("ERR:"): return r,[]
    try: d=json.loads(r)
    except Exception as e: return "PARSE:"+str(e)[:80],[]
    out=d.get("data") or d.get("results") or []
    tot=d.get("total")
    return "total=%s"%tot, out[:n]
def oa(q,n=12):
    u="https://api.openalex.org/works?search="+urllib.parse.quote(q)+"&per-page=%d&mailto=research@example.com"%n
    r=get(u)
    if r.startswith("ERR:"): return r,[]
    try: d=json.loads(r)
    except Exception as e: return "PARSE:"+str(e)[:80],[]
    return "count=%s"%d.get("meta",{}).get("count"), [{"title":w.get("title"),"year":w.get("publication_year"),"venue":(w.get("primary_location") or {}).get("source",{}).get("display_name") if (w.get("primary_location") or {}).get("source") else None,"doi":w.get("doi"),"abstract":None} for w in d.get("results",[])]
QUERIES=[
 "compressible natural convection surrogate model neural",
 "low Mach number natural convection machine learning",
 "non-Boussinesq natural convection neural network correction",
 "Boussinesq approximation breakdown correction data-driven",
 "pressure work viscous dissipation natural convection surrogate",
 "Mach number similarity scaling natural convection cavity",
]
log=open("lit/ma_screening.txt","w",encoding="utf-8")
for q in QUERIES:
    for name,fn in (("S2",s2),("OA",oa)):
        stat,rows=fn(q)
        print("="*100); print("[%s] %s  -> %s"%(name,q,stat))
        log.write("="*100+"\n[%s] %s -> %s\n"%(name,q,stat))
        for r in rows:
            t=(r.get("title") or "")[:150]; y=r.get("year"); v=(r.get("venue") or "")
            ids=r.get("externalIds") or {}
            doi=r.get("doi") or ids.get("DOI") or ""
            print("   - (%s) %s | %s | %s"%(y,t,v,doi))
            log.write("   - (%s) %s | %s | %s\n"%(y,t,v,doi))
        time.sleep(1.2)
log.close(); print("\nlog -> lit/ma_screening.txt")