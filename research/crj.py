import urllib.request, urllib.parse, json, re, time, os
OUT="research/lit"; os.makedirs(OUT, exist_ok=True)
def cr_journal(issn, q, rows=8):
    url = f"https://api.crossref.org/journals/{issn}/works?" + urllib.parse.urlencode(
        {"query.bibliographic": q, "rows": rows,
         "select":"DOI,title,container-title,issued,abstract,type,is-referenced-by-count"})
    req = urllib.request.Request(url, headers={"User-Agent":"lit-review/1.0 (mailto:lit@example.com)"})
    return json.load(urllib.request.urlopen(req, timeout=60))["message"]["items"]
JOURNALS = {
 "IJHMT (0017-9310)": "0017-9310",
 "IntCommHMT (0735-1933)": "0735-1933",
 "Applied Thermal Eng (1359-4311)": "1359-4311",
 "Int J Thermal Sci (1290-0729)": "1290-0729",
 "JQSRT (0022-4073)": "0022-4073",
}
QS = ["deep learning surrogate natural convection",
      "machine learning surrogate surface radiation convection",
      "multi-fidelity surrogate model correction heat transfer",
      "neural network extrapolation geometry scale heat transfer",
      "dimensional analysis machine learning dimensionless heat transfer"]
lines=[]
for jn, issn in JOURNALS.items():
    for q in QS:
        lines.append("="*110); lines.append(f"[{jn}] {q}")
        try: items = cr_journal(issn, q)
        except Exception as e:
            lines.append(f"  FAIL {e}"); items=[]
        for it in items:
            t=(it.get("title") or ["?"])[0]
            y=it.get("issued",{}).get("date-parts",[[None]])[0][0]
            lines.append(f"  - {it.get('DOI')} | {y} | cites={it.get('is-referenced-by-count')}")
            lines.append(f"      {t}")
            a=it.get("abstract")
            if a:
                a=" ".join(re.sub(r"<[^>]+>"," ",a).split())
                lines.append(f"      ABS: {a[:350]}")
        open(os.path.join(OUT,"journals.txt"),"w",encoding="utf-8").write("\n".join(lines))
        print(f"[{jn}] {q} -> {len(items)}", flush=True)
        time.sleep(1.0)
print("DONE")
