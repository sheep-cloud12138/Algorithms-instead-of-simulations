import requests, json, time
H={"User-Agent":"lit-scan/1.0"}
r=requests.get("https://api.openalex.org/works",headers=H,timeout=30,params={
 "search":"Benchmark solution 3D natural convection flows surface radiation air-filled cavities",
 "per-page":5,"select":"id,doi,title,publication_year,primary_location,authorships,locations,type"})
for w in r.json()["results"]:
    src=(w.get("primary_location") or {}).get("source") or {}
    print(f"\n{w['publication_year']} | type={w.get('type')} | {src.get('display_name')} | DOI={w.get('doi')}")
    print("  T:",w["title"])
    print("  auth:",[a["author"]["display_name"] for a in w.get("authorships",[])][:8])
    for l in (w.get("locations") or [])[:4]:
        s=(l.get("source") or {}) or {}
        print("   loc:",s.get("display_name"),"|",l.get("landing_page_url"))
print("\n--- 其他 benchmark 线索 ---")
r2=requests.get("https://api.openalex.org/works",headers=H,timeout=30,params={
 "filter":"title_and_abstract.search:benchmark coupled radiation convection cavity validation","per-page":8,
 "select":"doi,title,publication_year,primary_location"})
for w in r2.json()["results"]:
    src=(w.get("primary_location") or {}).get("source") or {}
    print(f"  {w['publication_year']} | {str(src.get('display_name'))[:38]:38s} | {str(w['title'])[:80]} | {w.get('doi')}")
