import sys, time, os, urllib.request, urllib.parse, xml.etree.ElementTree as ET
NS = {"a": "http://www.w3.org/2005/Atom"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lit")
os.makedirs(OUT, exist_ok=True)

def q(query, n=15, sort="relevance"):
    url = ("http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": n, "sortBy": sort}))
    for _ in range(4):
        try:
            raw = urllib.request.urlopen(url, timeout=45).read()
            break
        except Exception as e:
            print("   !! retry", e, flush=True); time.sleep(5)
    else:
        return None
    root = ET.fromstring(raw); out = []
    for e in root.findall("a:entry", NS):
        out.append((e.find("a:id", NS).text.split("/abs/")[-1],
                    e.find("a:published", NS).text[:10],
                    " ".join(e.find("a:title", NS).text.split()),
                    " ".join(e.find("a:summary", NS).text.split())))
    return out

def run(batch, queries, n=15, width=700):
    lines = []
    for label, query in queries:
        lines.append("=" * 110)
        lines.append(f"[{label}] QUERY: {query}")
        try:
            res = q(query, n)
        except Exception as e:
            res = None; lines.append("   (EXC %s)" % e)
        p = os.path.join(OUT, f"batch{batch}.txt")
        open(p, "w", encoding="utf-8").write("\n".join(lines))
        if res is None:
            lines.append("   (API FAILED)")
        elif not res:
            lines.append("   (no hits)")
        for aid, pub, title, summ in res:
            lines.append(f"  - {aid} {pub} {title}")
            lines.append(f"      {summ[:width]}")
        print(f"[{label}] {query} -> {len(res) if res else 0} hits", flush=True)
        time.sleep(3.5)
    p = os.path.join(OUT, f"batch{batch}.txt")
    open(p, "w", encoding="utf-8").write("\n".join(lines))
    print("WROTE", p, flush=True)

B = {
 "2": [
    ("Q2","abs:\"non-Boussinesq\" AND abs:\"machine learning\""),
    ("Q2","abs:\"non-Boussinesq\" AND abs:\"surrogate\""),
    ("Q2","abs:\"departure from Boussinesq\""),
    ("Q2","abs:\"low-Mach\" AND abs:\"machine learning\""),
    ("Q2","abs:\"Boussinesq approximation\" AND abs:\"neural operator\""),
    ("Q2","abs:\"variable properties\" AND abs:\"natural convection\" AND abs:\"machine learning\""),
    ("Q2","abs:\"similarity\" AND abs:\"breaking\" AND abs:\"correction\""),
    ("Q2","abs:\"scaling\" AND abs:\"correction\" AND abs:\"surrogate\" AND abs:\"convection\""),
 ],
 "3": [
    ("Q3","abs:\"surface radiation\" AND abs:\"natural convection\" AND abs:\"machine learning\""),
    ("Q3","abs:\"radiation\" AND abs:\"natural convection\" AND abs:\"surrogate\""),
    ("Q3","abs:\"participating media\" AND abs:\"neural operator\""),
    ("Q3","abs:\"radiative transfer\" AND abs:\"neural operator\""),
    ("Q3","abs:\"coupled\" AND abs:\"radiation\" AND abs:\"convection\" AND abs:\"deep learning\""),
    ("Q3","abs:\"surface-to-surface radiation\" AND abs:\"surrogate\""),
    ("Q3","abs:\"radiation\" AND abs:\"buoyancy\" AND abs:\"neural network\""),
 ],
 "4": [
    ("Q4","abs:\"neural operator\" AND abs:\"geometry generalization\""),
    ("Q4","abs:\"extrapolation\" AND abs:\"neural operator\" AND abs:\"scale\""),
    ("Q4","abs:\"Reynolds number\" AND abs:\"extrapolation\" AND abs:\"machine learning\""),
    ("Q4","abs:\"unseen\" AND abs:\"neural operator\" AND abs:\"training domain\""),
    ("Q4","abs:\"heat transfer\" AND abs:\"extrapolation\" AND abs:\"neural network\""),
    ("Q4","abs:\"domain of validity\" AND abs:\"surrogate\" AND abs:\"physics\""),
 ],
 "5": [
    ("Q5","abs:\"multi-fidelity\" AND abs:\"heat transfer\" AND abs:\"neural network\""),
    ("Q5","abs:\"multi-fidelity\" AND abs:\"CFD\" AND abs:\"correction\""),
    ("Q5","abs:\"low-fidelity\" AND abs:\"high-fidelity\" AND abs:\"convection\" AND abs:\"surrogate\""),
    ("Q5","abs:\"physics-based\" AND abs:\"correction\" AND abs:\"deep learning\" AND abs:\"turbulence\""),
    ("Q5","abs:\"baseline\" AND abs:\"correction\" AND abs:\"neural operator\" AND abs:\"PDE\""),
    ("Q5","abs:\"residual learning\" AND abs:\"physics solver\" AND abs:\"neural network\""),
 ],
}

B["6"] = [
    ("Q1","abs:\"physical units\" AND abs:\"neural operator\""),
    ("Q1","abs:\"scale equivariance\" AND abs:\"PDE\""),
    ("Q1","abs:\"unit-aware\" AND abs:\"neural network\""),
    ("Q1","abs:\"dimensionless numbers\" AND abs:\"deep learning\""),
    ("Q1","ti:\"dimensional\" AND abs:\"operator learning\""),
    ("Q1","abs:\"similarity\" AND abs:\"dimensionless\" AND abs:\"neural network\""),
    ("Q6","abs:\"exact symmetry\" AND abs:\"machine learning\" AND abs:\"physics\""),
    ("Q6","abs:\"group equivariant\" AND abs:\"neural operator\""),
    ("Q6","abs:\"Wigner\" AND abs:\"dimensional analysis\" AND abs:\"machine learning\""),
    ("Q6","abs:\"quantify\" AND abs:\"similarity solution\" AND abs:\"deep learning\""),
    ("Q6","abs:\"self-similar\" AND abs:\"neural network\" AND abs:\"convection\""),
    ("Q6","abs:\"thermal radiation\" AND abs:\"natural convection\" AND abs:\"CFD\" AND abs:\"cavity\""),
]

B["7"] = [
    ("Q2","abs:\"Boussinesq\" AND abs:\"correction\" AND abs:\"machine learning\""),
    ("Q2","abs:\"Boussinesq\" AND abs:\"compressible\" AND abs:\"surrogate\""),
    ("Q2","abs:\"model discrepancy\" AND abs:\"convection\""),
    ("Q3","abs:\"radiation\" AND abs:\"natural convection\" AND abs:\"neural network\""),
    ("Q3","abs:\"radiative\" AND abs:\"buoyant\" AND abs:\"learning\""),
    ("Q4","abs:\"geometric scaling\" AND abs:\"neural network\""),
    ("Q4","abs:\"scale generalization\" AND abs:\"neural operator\""),
    ("Q4","abs:\"trained on small\" AND abs:\"larger\""),
    ("Q4","abs:\"length scale\" AND abs:\"generalization\" AND abs:\"surrogate\""),
    ("Q5","abs:\"discrepancy\" AND abs:\"multi-fidelity\" AND abs:\"heat transfer\""),
    ("Q5","abs:\"analytic\" AND abs:\"correction\" AND abs:\"surrogate\" AND abs:\"heat transfer\""),
    ("Q5","abs:\"hybrid\" AND abs:\"low-fidelity solver\" AND abs:\"neural\""),
]

B["8"] = [
    ("Q1","abs:\"equivariant\" AND abs:\"units\""),
    ("Q1","abs:\"dimensional analysis\" AND abs:\"neural network\" AND abs:\"PDE\""),
    ("Q1","abs:\"nondimensionalization\" AND abs:\"neural\""),
    ("Q1","abs:\"physical units\" AND abs:\"physics-informed\""),
    ("Q3","abs:\"conjugate heat transfer\" AND abs:\"radiation\" AND abs:\"machine learning\""),
    ("Q3","abs:\"surface radiation\" AND abs:\"surrogate\" AND abs:\"heat\""),
    ("Q3","abs:\"natural convection\" AND abs:\"radiation\" AND abs:\"surrogate\" AND abs:\"enclosure\""),
    ("Q6","abs:\"similarity solution\" AND abs:\"neural network\""),
    ("Q6","abs:\"asymptotic\" AND abs:\"correction\" AND abs:\"neural network\" AND abs:\"convection\""),
    ("Q6","abs:\"self-similarity\" AND abs:\"deep learning\" AND abs:\"flow\""),
    ("Q6","abs:\"invariant\" AND abs:\"scaling\" AND abs:\"neural operator\""),
]
if __name__ == "__main__" and len(sys.argv) > 1:
    run(sys.argv[1], B[sys.argv[1]])
