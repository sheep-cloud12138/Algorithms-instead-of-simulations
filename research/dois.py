import urllib.request, urllib.parse, json, re, time, os
OUT="research/lit"
def doi_abs(doi):
    url="https://api.crossref.org/works/"+urllib.parse.quote(doi)
    req=urllib.request.Request(url, headers={"User-Agent":"lit-review/1.0 (mailto:lit@example.com)"})
    m=json.load(urllib.request.urlopen(req,timeout=50))["message"]
    a=m.get("abstract")
    if a: a=" ".join(re.sub(r"<[^>]+>"," ",a).split())
    return {"doi":m.get("DOI"),"t":(m.get("title") or ["?"])[0],"y":m.get("issued",{}).get("date-parts",[[None]])[0][0],
            "j":(m.get("container-title") or ["?"])[0] if m.get("container-title") else "?","a":a}
DOIS=["10.1016/j.ijheatmasstransfer.2023.124593",
      "10.1016/j.ijheatmasstransfer.2023.124336",
      "10.1016/j.ijheatmasstransfer.2024.126585",
      "10.1016/j.ijheatmasstransfer.2025.128205",
      "10.1016/j.ijheatmasstransfer.2022.123608",
      "10.1016/j.ijheatmasstransfer.2024.125441",
      "10.1016/j.applthermaleng.2024.124258",
      "10.1016/j.icheatmasstransfer.2024.107808",
      "10.1016/j.ijheatmasstransfer.2019.05.014",
      "10.1016/j.ijheatmasstransfer.2026.128921"]
out=[]
for d in DOIS:
    try: r=doi_abs(d)
    except Exception as e: r={"doi":d,"t":f"FAIL {e}","y":None,"j":None,"a":None}
    out.append("="*100)
    out.append(f"{r['doi']} | {r['y']} | {r['j']}")
    out.append(f"  {r['t']}")
    out.append(f"  ABS: {r['a']}")
    print(f"{r['doi']} ok", flush=True); time.sleep(0.8)
open(os.path.join(OUT,"doi_abs.txt"),"w",encoding="utf-8").write("\n".join(out))
print("DONE")
