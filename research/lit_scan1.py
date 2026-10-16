import requests, json, time, os
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
os.makedirs("lit",exist_ok=True)

def oa(q, filt=None, n=25):
    f=f"title_and_abstract.search:{q}"
    if filt: f=f+","+filt
    for attempt in range(3):
        try:
            r=requests.get("https://api.openalex.org/works",headers=H,timeout=30,params={
              "filter":f,"per-page":n,
              "select":"id,doi,title,publication_year,primary_location,abstract_inverted_index,cited_by_count,type"})
            if r.status_code==200: return r.json()
        except Exception: pass
        time.sleep(2)
    return None

def s2(q, n=100):
    for attempt in range(4):
        try:
            r=requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",headers=H,timeout=40,
              params={"query":q,"fields":"title,year,venue,externalIds,abstract,publicationTypes,citationCount"})
            if r.status_code==200: return r.json()
            if r.status_code==429: time.sleep(5+attempt*5); continue
            return {"err":r.status_code}
        except Exception: time.sleep(3)
    return {"err":"timeout"}

def epmc(q, n=50):
    for attempt in range(3):
        try:
            r=requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",headers=H,timeout=30,
              params={"query":q,"format":"json","pageSize":n,"resultType":"core"})
            if r.status_code==200: return r.json()
        except Exception: pass
        time.sleep(2)
    return None

Q1=["surface radiation natural convection surrogate",
    "surface radiation natural convection machine learning",
    "surface radiation natural convection neural network",
    "surface radiation natural convection reduced-order model",
    "radiative heat transfer natural convection surrogate model",
    "thermal radiation natural convection deep learning",
    "radiation coupled natural convection data-driven",
    "surface radiation buoyancy-driven cavity surrogate",
    "participating media natural convection surrogate",
    "radiation natural convection digital twin"]
Q2=["Stark number natural convection",
    "Planck number natural convection",
    "conduction radiation parameter natural convection",
    "radiation Nusselt number natural convection",
    "similarity radiation natural convection scaling",
    "scale analysis surface radiation cavity",
    "radiation effect similarity law deviation",
    "buoyancy radiation similarity breakdown"]
Q3=["multi-fidelity natural convection heat transfer",
    "multi-fidelity radiation heat transfer",
    "low-fidelity correction high-fidelity thermal",
    "model discrepancy correction conjugate heat transfer",
    "transfer learning low-fidelity CFD heat transfer",
    "physics-informed correction radiation"]
Q4=["review surface radiation natural convection enclosure",
    "benchmark natural convection radiation square cavity",
    "radiation convection interaction review"]

ALL=[("Q1",q) for q in Q1]+[("Q2",q) for q in Q2]+[("Q3",q) for q in Q3]+[("Q4",q) for q in Q4]
out={}
for tag,q in ALL:
    d=oa(q)
    c=d["meta"]["count"] if d else None
    out[q]={"openalex":d}
    print(f"[{tag}] OA={c:6} | {q}")
    time.sleep(0.3)
json.dump(out,open("lit/scan_openalex.json","w",encoding="utf-8"),ensure_ascii=False)
print("saved lit/scan_openalex.json")
