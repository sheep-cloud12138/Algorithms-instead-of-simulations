import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
def cr_abs(doi):
    r=requests.get(f"https://api.crossref.org/works/{doi}",headers=H,timeout=30)
    if r.status_code!=200: return None
    return r.json()["message"]
for d in ["10.1017/jfm.2020.395","10.1016/J.ENCONMAN.2013.01.024","10.1016/J.IJHEATMASSTRANSFER.2016.11.061"]:
    m=cr_abs(d)
    print(f"\n===== {d} =====")
    if not m: print("  crossref fail"); continue
    print("  T:",(m.get("title") or [""])[0])
    print("  J:",(m.get("container-title") or [""])[0],"|",(m.get("issued",{}).get("date-parts") or [[None]])[0][0])
    a=m.get("abstract")
    import re
    if a: print("  ABS:", re.sub("<[^>]+>","",a)[:1400])
    else: print("  ABS: (crossref 无)")
    time.sleep(0.5)
