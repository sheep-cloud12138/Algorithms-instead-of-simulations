import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
QS=["surface radiation natural convection surrogate model",
 "radiation natural convection machine learning surrogate",
 "radiation coupled natural convection reduced order model",
 "multi-fidelity natural convection radiation heat transfer",
 "radiation convection similarity scaling parameter",
 "surface radiation natural convection benchmark cavity",
 "radiation convection interaction review",
 "machine learning surrogate coupled radiation convection"]
out={}
for q in QS:
    for a in range(3):
        try:
            r=requests.get("https://api.crossref.org/works",headers=H,timeout=30,params={
              "query.bibliographic":q,"rows":20,
              "select":"DOI,title,container-title,issued,type,abstract"})
            if r.status_code==200: break
        except Exception: pass
        time.sleep(2)
    d=r.json()["message"]
    print(f"\n### Crossref total={d['total-results']} | {q}")
    out[q]=d
    for it in d["items"][:12]:
        ct=(it.get("container-title") or [""])[0]
        yr=(it.get("issued",{}).get("date-parts") or [[None]])[0][0]
        print(f"  {yr} | {str(ct)[:36]:36s} | {str((it.get('title') or [''])[0])[:84]} | {it['DOI']}")
    time.sleep(0.6)
json.dump(out,open("lit/scan_crossref.json","w",encoding="utf-8"),ensure_ascii=False)
print("\nsaved")
