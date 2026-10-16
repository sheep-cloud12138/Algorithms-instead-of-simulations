import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
d=json.load(open("lit/scan_openalex.json",encoding="utf-8"))
watch=[q for q in d if any(k in q for k in [
 "surrogate model","machine learning","neural network","reduced-order model","deep learning",
 "data-driven","digital twin","Stark number","similarity law deviation","similarity radiation",
 "model discrepancy","review surface radiation","benchmark natural convection radiation",
 "buoyancy-driven cavity surrogate","participating media"])]
seen={}
for q in watch:
    r=d[q]["openalex"]["results"]
    print(f"\n=== {q}  (count={d[q]['openalex']['meta']['count']}) ===")
    for w in r:
        src=(w.get("primary_location") or {}).get("source") or {}
        t=w["title"] or ""
        key=(w["doi"] or w["id"])
        print(f"  {w['publication_year']} | {str(src.get('display_name'))[:38]:38s} | {t[:95]}")
        seen[key]=(w["publication_year"],str(src.get('display_name')),t,w.get("doi"))
print(f"\n\nTOTAL UNIQUE: {len(seen)}")
json.dump(seen,open("lit/oa_q1hits.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
