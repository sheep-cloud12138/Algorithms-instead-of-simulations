import requests, json, time, re
H={"User-Agent":"lit-scan/1.0"}
DOIS=["10.1016/j.ijheatmasstransfer.2026.129230","10.1016/j.icheatmasstransfer.2025.109805",
      "10.1016/J.IJHEATMASSTRANSFER.2016.11.061","10.1108/sr-07-2025-0509","10.1108/hff-04-2026-0491"]
print("=========== OpenAIRE ===========")
for d in DOIS:
    try:
        r=requests.get("https://api.openaire.eu/search/publications",headers=H,timeout=40,params={"doi":d,"format":"json"})
        txt=r.text
        m=re.search(r'"description"[^"]*"\s*:\s*(\{.*?\}|\[.*?\]|"[^"]{0,900}")', txt, re.S)
        print(f"\n### {d} | {r.status_code} | {len(txt)}B")
        if m: print("  ", re.sub(r"\s+"," ",m.group(1))[:700])
        else: print("   (未解析到 description)")
    except Exception as e:
        print(f"\n### {d} FAIL {type(e).__name__}")
    time.sleep(1)
print("\n=========== S2 补充 ===========")
for d in ["10.1108/sr-07-2025-0509","10.1108/hff-04-2026-0491","10.1108/hff-04-2026-0491"]:
    for a in range(4):
        r=requests.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{d}",headers=H,timeout=40,
          params={"fields":"title,year,venue,abstract"})
        if r.status_code==200: break
        if r.status_code==429: time.sleep(6+a*6); continue
        break
    if r.status_code!=200: print(f"  {d} ERR {r.status_code}"); time.sleep(2); continue
    p=r.json(); print(f"\n### {d}\n  {p.get('year')} | {p.get('venue')}\n  T: {p.get('title')}\n  A: {(p.get('abstract') or '(无)')[:800]}")
    time.sleep(3.5)
