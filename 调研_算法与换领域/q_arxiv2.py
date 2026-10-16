# -*- coding: utf-8 -*-
import json, os, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
NS = {"a": "http://www.w3.org/2005/Atom"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

def ax(search, n=15):
    url = ("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(search)
           + f"&start=0&max_results={n}&sortBy=relevance&sortOrder=descending")
    for a in range(3):
        try:
            root = ET.fromstring(urllib.request.urlopen(url, timeout=40).read())
            return [{"id": e.findtext("a:id","",NS).split("/abs/")[-1],
                     "t": " ".join(e.findtext("a:title","",NS).split())[:150],
                     "y": e.findtext("a:published","",NS)[:7]} for e in root.findall("a:entry", NS)]
        except Exception as ex:
            if a == 2: return [{"err": str(ex)}]
            time.sleep(3)

Q = [
 ("scalinglaw_disc", 'abs:"discovering scaling laws" OR abs:"data-driven scaling law" OR ti:"scaling law" AND abs:"symbolic regression"'),
 ("cross_phys",      'ti:"Cross-Physics" OR abs:"cross-physics mapping"'),
 ("correction_law",  'abs:"correction" AND abs:"empirical correlation" AND abs:"neural"'),
 ("knudsen_surr",    'abs:"Knudsen" AND abs:"surrogate"'),
 ("stefan_ml",       'abs:"Stefan problem" AND abs:"machine learning"'),
 ("stefan_nn",       'abs:"phase change" AND abs:"similarity solution"'),
 ("size_effect_ml",  'abs:"size effect" AND abs:"machine learning"'),
 ("nonnewton_ml",    'abs:"non-Newtonian" AND abs:"surrogate"'),
 ("marangoni_ml",    'abs:"Marangoni" AND abs:"surrogate" OR abs:"Marangoni" AND abs:"neural network"'),
 ("ehl_ml",          'abs:"elastohydrodynamic" AND abs:"machine learning"'),
 ("forchheimer_ml",  'abs:"Forchheimer" AND abs:"neural"'),
 ("multifid_law",    'abs:"multi-fidelity" AND abs:"scaling" AND abs:"physical model"'),
 ("nonlocal_ml",     'abs:"nonlocal" AND abs:"constitutive" AND abs:"neural network"'),
]

res = {}
for name, s in Q:
    r = ax(s)
    res[name] = {"q": s, "hits": r}
    print("\n### %-16s %s" % (name, s))
    for h in r[:9]:
        if "err" in h: print("   ERR", h["err"]); continue
        print("   (%s) %s | %s" % (h["y"], h["id"], h["t"]))
    time.sleep(3)
json.dump(res, open(os.path.join(OUT,"arxiv2.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
