import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
J={"IJHMT":"S90999920","ATE":"S129676613","IJTS":"S128740003","ICHMT":"S118246204","JQSRT":"S100373537"}
def oa(q,filt=None,n=25,sel="id,doi,title,publication_year,primary_location,cited_by_count"):
    f=f"title_and_abstract.search:{q}"
    if filt: f+=","+filt
    for a in range(3):
        r=requests.get("https://api.openalex.org/works",headers=H,timeout=30,params={"filter":f,"per-page":n,"select":sel})
        if r.status_code==200: return r.json()
        time.sleep(2)
    return None
print("### 期刊限定: radiation + natural convection + (surrogate|ML|neural|reduced-order), 2020-2026")
for jn,jid in J.items():
    for q in ['"surface radiation" "natural convection" surrogate', '"radiation" "natural convection" "machine learning"',
              '"radiation" "natural convection" "neural network"']:
        d=oa(q,f"primary_location.source.id:{jid},from_publication_date:2020-01-01")
        if not d: print("  fail",jn); continue
        if d["meta"]["count"]>0:
            print(f"  [{jn}] count={d['meta']['count']} <- {q}")
            for w in d["results"][:8]:
                print(f"      {w['publication_year']} | {str(w['title'])[:90]} | {w.get('doi')}")
        time.sleep(0.3)
print("\n### 期刊限定: 近3年 辐射+对流 全部(不限方法)，看有无代理模型")
for jn in ["IJHMT","ATE","IJTS"]:
    d=oa('"radiation" "natural convection"',f"primary_location.source.id:{J[jn]},from_publication_date:2023-01-01",n=50)
    print(f"  [{jn}] count={d['meta']['count']}")
