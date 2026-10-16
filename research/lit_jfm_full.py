import requests, json, time
H={"User-Agent":"lit-scan/1.0 (mailto:researcher@example.com)"}
def inv2abs(inv):
    if not inv: return None
    pos={}
    for w,ps in inv.items():
        for p in ps: pos[p]=w
    return " ".join(pos[k] for k in sorted(pos))
r=requests.get("https://api.openalex.org/works/https://doi.org/10.1017/jfm.2020.395",headers=H,timeout=30).json()
print("=== JFM 2020.395 FULL ABSTRACT ===")
print(inv2abs(r.get("abstract_inverted_index")))
print("\nAUTHORS:", [a["author"]["display_name"] for a in r.get("authorships",[])])
print("INST:", list({i["display_name"] for a in r.get("authorships",[]) for i in a.get("institutions",[])}))
print("CONCEPTS:", [c["display_name"] for c in r.get("concepts",[])[:8]])
