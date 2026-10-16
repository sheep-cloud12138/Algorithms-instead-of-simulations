import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
r=requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",headers=H,timeout=30,
  params={"query":"surface radiation natural convection cavity","fields":"title,year,venue,externalIds,abstract"})
print("status",r.status_code)
print(r.text[:800])
