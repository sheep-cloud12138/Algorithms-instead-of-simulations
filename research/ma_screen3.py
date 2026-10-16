import urllib.request,json,urllib.parse,time,io,sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
UA={"User-Agent":"Mozilla/5.0 (research)"}
def oa(q,n=20):
    u="https://api.openalex.org/works?filter="+urllib.parse.quote(q,safe=':,')+"&per-page=%d&mailto=r@example.com&select=title,publication_year,doi,primary_location,abstract_inverted_index"%n
    try: d=json.loads(urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60).read().decode("utf-8","replace"))
    except Exception as e: return "ERR:"+str(e)[:100],[]
    out=[]
    for w in d.get("results",[]):
        src=(w.get("primary_location") or {}).get("source") or {}
        ab=w.get("abstract_inverted_index"); txt=""
        if ab:
            pos={}
            for word,ii in ab.items():
                for i in ii: pos[i]=word
            txt=" ".join(pos[k] for k in sorted(pos))[:300]
        out.append((w.get("publication_year"),w.get("title"),src.get("display_name"),w.get("doi"),txt))
    return d.get("meta",{}).get("count"),out
Q=[('title_and_abstract.search:non-Boussinesq AND title_and_abstract.search:"machine learning" OR title_and_abstract.search:non-Boussinesq AND title_and_abstract.search:surrogate','NOB/非Boussinesq + ML'),
   ('title_and_abstract.search:"neural operator" AND title_and_abstract.search:Boussinesq','神经算子 + Boussinesq'),
   ('title_and_abstract.search:"Fourier neural operator" AND title_and_abstract.search:convection AND title_and_abstract.search:cavity','FNO + 腔体对流'),
   ('title_and_abstract.search:"viscous dissipation" AND title_and_abstract.search:"natural convection" AND title_and_abstract.search:neural','耗散 + 自然对流 + 神经'),
   ('title_and_abstract.search:"multi-fidelity" AND title_and_abstract.search:"natural convection"','多保真 + 自然对流'),
   ('title_and_abstract.search:"departure from the Boussinesq" OR title_and_abstract.search:"breakdown of the Boussinesq"','Boussinesq 失效/破缺'),
   ('title_and_abstract.search:surrogate AND title_and_abstract.search:"Rayleigh-Benard" AND title_and_abstract.search:neural','RB 对流代理')]
for q,lab in Q:
    c,rows=oa(q)
    print("="*110); print("QUERY[%s] -> count=%s"%(lab,c))
    for y,t,v,doi,ab in rows[:10]:
        print("   - (%s) %s | %s | %s"%(y,t,v,doi))
        if ab: print("       ABS: %s"%ab[:230])
    time.sleep(0.7)