import requests, json, time
H={"User-Agent":"lit-scan/1.0"}
def s2(q):
    for a in range(5):
        try:
            r=requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",headers=H,timeout=45,
              params={"query":q,"fields":"title,year,venue,externalIds"})
            if r.status_code==200: return r.json()
            if r.status_code==429: time.sleep(6+a*6); continue
            return {"err":r.status_code}
        except Exception: time.sleep(4)
    return {"err":"fail"}
QS=["model form error machine learning missing physics radiation",
 "data-driven closure radiation heat transfer correction",
 "Fourier neural operator radiative convective heat transfer",
 "DeepONet radiation convection heat transfer",
 "surrogate model radiation participating medium natural convection cavity",
 "machine learning Nusselt number correlation surface radiation enclosure",
 "emulator radiative transfer coupled convection heat transfer",
 "neural network missing physics correction thermal simulation",
 "opacity radiation reduced order model convection"]
out={}
for q in QS:
    d=s2(q); print(f"\n### total={d.get('total')} err={d.get('err')} | {q}")
    for p in (d.get("data") or [])[:8]:
        ex=p.get("externalIds") or {}
        print(f"  {p.get('year')} | {str(p.get('venue'))[:32]:32s} | {str(p.get('title'))[:82]}")
        if ex.get("DOI"): print(f"        DOI:{ex['DOI']}")
    out[q]=d; time.sleep(3.5)
json.dump(out,open("lit/scan_s2c.json","w",encoding="utf-8"),ensure_ascii=False)
print("\nsaved")
