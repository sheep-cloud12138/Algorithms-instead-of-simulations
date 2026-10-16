import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}

# 1) OpenAlex 是否带 Elsevier 摘要？
r=requests.get("https://api.openalex.org/works",headers=H,timeout=30,params={
 "filter":"title_and_abstract.search:surface radiation natural convection",
 "per-page":5,"select":"id,doi,title,publication_year,primary_location,abstract_inverted_index"})
d=r.json()
print("OpenAlex hits:",d["meta"]["count"])
for w in d["results"][:5]:
    src=(w.get("primary_location") or {}).get("source") or {}
    print("  -",w["publication_year"],src.get("display_name"),"| abs?",bool(w.get("abstract_inverted_index")),"|",w["title"][:80])

# 2) S2 bulk 默认搜什么？已知 IJHMT 辐射对流论文测试
q="surface radiation natural convection cavity"
r=requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",headers=H,timeout=30,
  params={"query":q,"fields":"title,year,venue,externalIds,abstract"})
d=r.json()
print("\nS2 bulk total:",d.get("total"))
for p in (d.get("data") or [])[:6]:
    print("  -",p.get("year"),p.get("venue"),"| abs?",bool(p.get("abstract")),"|",str(p.get("title"))[:80])
