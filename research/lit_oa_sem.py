import requests, json
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
tests=[
 "surface radiation natural convection surrogate",
 "surface radiation natural convection",
 "radiation natural convection machine learning",
 "surface radiation natural convection reduced-order model",
]
for q in tests:
    r=requests.get("https://api.openalex.org/works",headers=H,timeout=30,params={
      "filter":f"title_and_abstract.search:{q}","per-page":1,"select":"id"})
    print(f"{r.json()['meta']['count']:8d}  <-  {q}")
