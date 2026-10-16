# -*- coding: utf-8 -*-
import json, os, sys, time, urllib.parse, urllib.request
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

def s2(query, year="2010-2026", tries=5):
    u = ("https://api.semanticscholar.org/graph/v1/paper/search/bulk?query="
         + urllib.parse.quote(query) + f"&fields=title,year,venue&year={year}")
    for a in range(tries):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent":"s/1.0"}), timeout=45))
            return d
        except Exception as e:
            if a == tries-1: return {"error": str(e)}
            time.sleep(6 * (a+1))

Q = [
 ("sco2_htd_ml",  '"heat transfer deterioration" "machine learning"'),
 ("sco2_corr_ml", '"supercritical" "heat transfer correlation" neural'),
 ("dittus_corr",  '"Dittus-Boelter"'),
 ("dittus_ml",    '"Dittus-Boelter" neural network'),
 ("chf_ml",       '"critical heat flux" "machine learning"'),
 ("pool_boil_ml", '"pool boiling" "machine learning"'),
 ("condens_ml",   '"condensation heat transfer" "machine learning"'),
 ("stefan_ml",    '"Stefan problem" "machine learning"'),
 ("stefan_pcm_ml",'"phase change" "similarity solution" machine learning'),
 ("rarefied_sr",  '"rarefied gas" surrogate'),
 ("knudsen_ml",   '"Knudsen number" "machine learning"'),
 ("nonfourier_ml",'"non-Fourier" heat conduction "machine learning"'),
 ("sizeeffect_ml",'"size effect" fracture "machine learning"'),
 ("ehl_ml",       '"elastohydrodynamic lubrication" "machine learning"'),
 ("nondarcy_ml",  '"non-Darcy" "neural network"'),
 ("marangoni_ml", '"Marangoni convection" "machine learning"'),
 ("mixedconv_ml", '"mixed convection" "machine learning"'),
 ("varprop_ml",   '"variable properties" "natural convection" "machine learning"'),
 ("porous_conv_ml",'"porous medium" "natural convection" "machine learning"'),
 ("simlaw_ml",    '"similarity law" "machine learning"'),
 ("exp_recover",  'discovering scaling exponents from data neural network'),
 ("corr_correct", 'machine learning correction empirical correlation heat transfer'),
 ("lowfi_correct_ht",'low-fidelity high-fidelity correction heat transfer surrogate'),
]
res = {}
for name, term in Q:
    d = s2(term)
    if "error" in d:
        print("[ERR ] %-18s %s -> %s" % (name, term, d["error"])); res[name]={"err":d["error"]}; time.sleep(4); continue
    tot = d.get("total", 0); data = d.get("data") or []
    res[name] = {"q": term, "total": tot, "top": [{"y":p.get("year"),"t":p.get("title"),"v":p.get("venue")} for p in data[:8]]}
    print("[%5s] %-18s %s" % (tot, name, term))
    for p in data[:5]:
        print("        - (%s) %s | %s" % (p.get("year"), p.get("title"), p.get("venue")))
    time.sleep(4)
json.dump(res, open(os.path.join(OUT,"s2_gap.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved raw/s2_gap.json")
