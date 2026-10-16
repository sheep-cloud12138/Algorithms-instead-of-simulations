import urllib.request,json,urllib.parse,time,io,sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
UA={"User-Agent":"Mozilla/5.0 (research)"}
def oa(parts,n=20,tries=3):
    f=",".join('title_and_abstract.search:"%s"'%p for p in parts)
    u="https://api.openalex.org/works?filter="+urllib.parse.quote(f,safe=':,')+"&per-page=%d&mailto=r@example.com&select=title,publication_year,doi,primary_location,abstract_inverted_index"%n
    for k in range(tries):
        try:
            d=json.loads(urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60).read().decode("utf-8","replace")); break
        except Exception as e:
            if k==tries-1: return "ERR:"+str(e)[:90],[]
            time.sleep(3)
    out=[]
    for w in d.get("results",[]):
        src=(w.get("primary_location") or {}).get("source") or {}
        ab=w.get("abstract_inverted_index"); txt=""
        if ab:
            pos={}
            for word,ii in ab.items():
                for i in ii: pos[i]=word
            txt=" ".join(pos[k] for k in sorted(pos))[:280]
        out.append((w.get("publication_year"),w.get("title"),src.get("display_name"),w.get("doi"),txt))
    return d.get("meta",{}).get("count"),out
Q=[(["Boussinesq","neural operator"],"Boussinesq + 神经算子"),
   (["Boussinesq","surrogate model"],"Boussinesq + 代理模型"),
   (["Boussinesq","data-driven"],"Boussinesq + 数据驱动"),
   (["natural convection","multi-fidelity"],"自然对流 + 多保真"),
   (["Rayleigh-Benard","neural network"],"RB 对流 + 神经网络"),
   (["non-Boussinesq","neural"],"非Boussinesq + neural"),
   (["compressible","Rayleigh-Benard","machine learning"],"可压缩 RB + ML")]
for parts,lab in Q:
    c,rows=oa(parts)
    print("="*110); print("QUERY[%s] -> count=%s"%(lab,c))
    for y,t,v,doi,ab in rows[:10]:
        print("   - (%s) %s | %s | %s"%(y,t,v,doi))
        if ab: print("       ABS: %s"%ab[:230])
    time.sleep(0.6)