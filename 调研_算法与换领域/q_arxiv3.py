# -*- coding: utf-8 -*-
import json, os, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
NS={"a":"http://www.w3.org/2005/Atom"}
def ax(s,n=10):
    u=("http://export.arxiv.org/api/query?search_query="+urllib.parse.quote(s)
       +f"&start=0&max_results={n}&sortBy=relevance&sortOrder=descending")
    for a in range(3):
        try:
            r=ET.fromstring(urllib.request.urlopen(u,timeout=40).read())
            return [{"id":e.findtext("a:id","",NS).split("/abs/")[-1],
                     "t":" ".join(e.findtext("a:title","",NS).split())[:140],
                     "y":e.findtext("a:published","",NS)[:7]} for e in r.findall("a:entry",NS)]
        except Exception as ex:
            if a==2: return [{"err":str(ex)}]
            time.sleep(3)
Q=[
 ("sco2_htd",     'abs:"heat transfer deterioration" OR abs:"supercritical" AND abs:"heat transfer" AND abs:"neural"'),
 ("sco2_ml2",     'abs:"supercritical carbon dioxide" AND abs:"machine learning"'),
 ("htd_corr_brk", 'abs:"Dittus-Boelter" OR abs:"heat transfer correlation" AND abs:"neural network"'),
 ("chf_ml",       'abs:"critical heat flux" AND abs:"machine learning"'),
 ("boiling_ml",   'abs:"flow boiling" AND abs:"machine learning"'),
 ("condens_ml",   'abs:"condensation heat transfer" AND abs:"machine learning"'),
 ("drying_ml",    'abs:"drying" AND abs:"porous" AND abs:"machine learning"'),
 ("simlaw_ml",    'abs:"similarity solution" AND abs:"neural network"'),
 ("law_break",    'abs:"departure from" AND abs:"scaling" AND abs:"neural"'),
 ("foundation_pde",'ti:"foundation model" AND abs:"partial differential equation"'),
 ("fm_survey",    'ti:"foundation models" AND abs:"simulation"'),
 ("sci_llm_agent",'abs:"large language model" AND abs:"computational fluid dynamics" AND abs:"agent"'),
]
for name,s in Q:
    r=ax(s); print("\n### %-14s %s" % (name,s))
    for h in r[:8]:
        if "err" in h: print("   ERR",h["err"]); continue
        print("   (%s) %s | %s" % (h["y"],h["id"],h["t"]))
    time.sleep(3)
