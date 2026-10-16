# -*- coding: utf-8 -*-
"""arXiv API: 近期预印本扫描 (OpenAlex 收录滞后)"""
import json, os, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
NS = {"a": "http://www.w3.org/2005/Atom"}

def ax(search, max_results=12, sort="relevance"):
    url = ("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(search)
           + f"&start=0&max_results={max_results}&sortBy={sort}&sortOrder=descending")
    for a in range(4):
        try:
            raw = urllib.request.urlopen(url, timeout=40).read()
            root = ET.fromstring(raw)
            out = []
            for e in root.findall("a:entry", NS):
                out.append({
                    "id": e.findtext("a:id", "", NS).split("/abs/")[-1],
                    "t": " ".join(e.findtext("a:title", "", NS).split())[:160],
                    "y": e.findtext("a:published", "", NS)[:7],
                    "c": (e.findtext("a:comment", "", NS) or "").replace("\n", " ")[:90],
                })
            return out
        except Exception as ex:
            if a == 3: return [{"err": str(ex)}]
            time.sleep(2 * (a + 1))

Q = [
 ("ax_nob_ml",      'abs:"non-Oberbeck-Boussinesq" OR abs:"non-Boussinesq"'),
 ("ax_bouss_corr",  'abs:"Boussinesq" AND abs:"correction"'),
 ("ax_lowfi_corr",  'abs:"low-fidelity" AND abs:"correction" AND abs:"neural"'),
 ("ax_multifid_hx", 'abs:"multi-fidelity" AND abs:"convection"'),
 ("ax_dimless",     'abs:"dimensional analysis" AND abs:"neural"'),
 ("ax_units_eq",    'abs:"units equivariance" OR abs:"dimensionless" AND abs:"neural operator"'),
 ("ax_extrap_no",   'abs:"neural operator" AND abs:"extrapolation"'),
 ("ax_fm_pde",      'abs:"foundation model" AND abs:"PDE"'),
 ("ax_scaling_law", 'abs:"scaling law" AND abs:"neural operator"'),
 ("ax_active_surr", 'abs:"active learning" AND abs:"surrogate" AND abs:"simulation"'),
 ("ax_rom_llm",     'abs:"reduced order" AND abs:"large language model"'),
 ("ax_knudsen_ml",  'abs:"rarefied" AND abs:"neural"'),
 ("ax_nonfourier",  'abs:"non-Fourier" AND abs:"neural"'),
 ("ax_phasefield_ml",'abs:"phase field" AND abs:"neural operator" OR abs:"phase field" AND abs:"machine learning"'),
 ("ax_peukert",     'abs:"battery" AND abs:"physics-informed"'),
]

if __name__ == "__main__":
    res = {}
    for name, s in Q:
        r = ax(s)
        res[name] = {"q": s, "n": len(r) if not (r and "err" in r[0]) else -1, "hits": r}
        print("\n### %s  [%s]  -> %d" % (name, s, len(r)))
        for h in r[:8]:
            if "err" in h: print("   ERR", h["err"]); continue
            print("   (%s) %s | %s | %s" % (h["y"], h["id"], h["t"], h["c"]))
        time.sleep(3)
    json.dump(res, open(os.path.join(OUT, "arxiv.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved raw/arxiv.json")
