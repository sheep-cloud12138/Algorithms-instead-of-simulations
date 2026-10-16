# -*- coding: utf-8 -*-
"""OpenAlex title+abstract 检索: 统计各方向论文密度 + 取头部文献"""
import json, sys, time, urllib.parse, urllib.request, os

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

MAIL = "survey@example.com"
BASE = "https://api.openalex.org/works"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

def q(term, per_page=8, sort="relevance_score:desc", extra="", year_from=None):
    f = f"title_and_abstract.search:{urllib.parse.quote(term)}"
    if year_from:
        f += f",from_publication_date:{year_from}-01-01"
    if extra:
        f += "," + extra
    url = f"{BASE}?filter={f}&per_page={per_page}&sort={sort}&mailto={MAIL}"
    req = urllib.request.Request(url, headers={"User-Agent": "survey/1.0"})
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=40))
            return d
        except Exception as e:
            if attempt == 3:
                return {"error": str(e)}
            time.sleep(2 * (attempt + 1))

def slim(w):
    return {
        "t": (w.get("title") or "")[:150],
        "y": w.get("publication_year"),
        "c": w.get("cited_by_count"),
        "v": (((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "")[:60],
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
    }

def run(queries, fname, per_page=6):
    res = {}
    print(f"\n{'='*90}\nFILE: {fname}  ({len(queries)} queries)\n{'='*90}")
    for name, term, year in queries:
        d = q(term, per_page=per_page, year_from=year)
        if "error" in d:
            res[name] = {"term": term, "error": d["error"]}
            print(f"[ERR ] {name:52s} {term}  -> {d['error']}")
            continue
        cnt = d["meta"]["count"]
        hits = [slim(w) for w in d["results"]]
        res[name] = {"term": term, "year_from": year, "count": cnt, "top": hits}
        print(f"[{cnt:>7}] {name:52s} {term}" + (f"  (>= {year})" if year else ""))
        for h in hits[:3]:
            print("          - (%s) %s  [%s] c=%s" % (h["y"], h["t"], h["v"], h["c"]))
        time.sleep(0.35)
    with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res
