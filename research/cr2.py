import urllib.request, urllib.parse, json, textwrap
def cr(q, rows=8, filt=None):
    p = {"query.bibliographic": q, "rows": rows, "select": "DOI,title,container-title,issued,abstract"}
    if filt: p["filter"] = filt
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers={"User-Agent":"lit-review/1.0 (mailto:student@example.edu)"})
    d = json.load(urllib.request.urlopen(req, timeout=45))
    return d["message"]["items"]
for q in ["surrogate model surface radiation coupled natural convection machine learning",
          "neural network coupled radiation convection heat transfer surrogate",
          "machine learning surrogate model natural convection enclosure"]:
    print("="*100); print("CROSSREF:", q)
    for it in cr(q):
        t = (it.get("title") or ["?"])[0]
        j = (it.get("container-title") or ["?"])[0]
        y = it.get("issued",{}).get("date-parts",[[None]])[0][0]
        print(f"  - {it.get('DOI')} | {y} | {j}")
        print(f"      {t}")
        a = it.get("abstract")
        if a:
            import re
            a = re.sub(r"<[^>]+>"," ",a)
            print("      ABS:", " ".join(a.split())[:300])
