import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
def s2(q):
    for a in range(6):
        try:
            r=requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",headers=H,timeout=45,
              params={"query":q,"fields":"title,year,venue,externalIds,abstract,publicationTypes,citationCount"})
            if r.status_code==200: return r.json()
            if r.status_code==429: time.sleep(6+a*6); continue
            return {"err":r.status_code,"body":r.text[:200]}
        except Exception as e:
            time.sleep(4)
    return {"err":"fail"}
QS=["surface radiation natural convection surrogate",
    "surface radiation natural convection machine learning",
    "radiation natural convection reduced-order model",
    "radiative heat transfer natural convection neural network",
    "thermal radiation natural convection deep learning surrogate",
    "radiation coupled natural convection data-driven model",
    "multi-fidelity radiation heat transfer",
    "multi-fidelity natural convection surrogate",
    "Stark number natural convection",
    "Planck number natural convection radiation",
    "conduction radiation parameter natural convection",
    "surface radiation natural convection square cavity benchmark",
    "radiation convection interaction review heat transfer",
    "model discrepancy low fidelity correction heat transfer",
    "low fidelity correction high fidelity thermal radiation"]
out={}
for q in QS:
    d=s2(q)
    tot=d.get("total"); err=d.get("err")
    print(f"S2 total={tot} err={err} | {q}")
    out[q]=d
    time.sleep(3.5)
json.dump(out,open("lit/scan_s2.json","w",encoding="utf-8"),ensure_ascii=False)
print("saved")
