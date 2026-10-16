import requests, json, time
H={"User-Agent":"lit-scan/1.0"}
def s2(q):
    for a in range(6):
        try:
            r=requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",headers=H,timeout=45,
              params={"query":q,"fields":"title,year,venue,externalIds,abstract,publicationTypes,citationCount"})
            if r.status_code==200: return r.json()
            if r.status_code==429: time.sleep(6+a*6); continue
            return {"err":r.status_code}
        except Exception: time.sleep(4)
    return {"err":"fail"}
QS=["surrogate model radiative convective heat transfer enclosure",
    "reduced order model natural convection surface radiation cavity",
    "proper orthogonal decomposition radiation natural convection",
    "discrete ordinates surrogate model radiation",
    "deep learning radiation heat transfer cavity prediction",
    "conjugate heat transfer radiation machine learning surrogate",
    "neural network emulator radiative transfer convection coupled",
    "radiation conduction parameter similarity scaling law",
    "Nusselt number correlation surface radiation natural convection enclosure",
    "machine learning correction CFD low fidelity model",
    "multi-fidelity surrogate thermal radiation conduction",
    "surrogate model participation media radiation convection"]
out={}
for q in QS:
    d=s2(q); print(f"S2 total={d.get('total')} err={d.get('err')} | {q}")
    out[q]=d; time.sleep(3.5)
json.dump(out,open("lit/scan_s2b.json","w",encoding="utf-8"),ensure_ascii=False)
print("saved")
