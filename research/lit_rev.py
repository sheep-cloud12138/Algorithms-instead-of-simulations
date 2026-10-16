import requests, json, time
H={"User-Agent":"lit-scan/1.0"}
J={"IJHMT":"S90999920","ATE":"S129676613","IJTS":"S128740003"}
base="from_publication_date:2023-01-01"
for jn,jid in J.items():
    for q in ['"radiation" "natural convection"','"surface radiation"','"thermal radiation" "convection"']:
        r=requests.get("https://api.openalex.org/works",headers=H,timeout=30,params={
         "filter":f"title_and_abstract.search:{q},primary_location.source.id:{jid},{base},type:review",
         "per-page":10,"select":"doi,title,publication_year"})
        d=r.json()
        if d["meta"]["count"]:
            print(f"\n[{jn}] review count={d['meta']['count']} <- {q}")
            for w in d["results"][:10]:
                print(f"   {w['publication_year']} | {str(w['title'])[:100]} | {w.get('doi')}")
        time.sleep(0.3)
print("\n=== IJHMT/ATE 2023+ 辐射+对流 标题一览(IJHMT) ===")
r=requests.get("https://api.openalex.org/works",headers=H,timeout=30,params={
 "filter":f"title_and_abstract.search:radiation natural convection,primary_location.source.id:S90999920,{base}",
 "per-page":30,"select":"doi,title,publication_year"})
for w in r.json()["results"]:
    print(f"  {w['publication_year']} | {str(w['title'])[:105]}")
