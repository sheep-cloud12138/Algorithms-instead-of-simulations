import os
p="research/lit_review.py"; s=open(p,encoding="utf-8").read()
extra = '''\nB["7"] = [
    ("Q2","abs:\\"Boussinesq\\" AND abs:\\"correction\\" AND abs:\\"machine learning\\""),
    ("Q2","abs:\\"Boussinesq\\" AND abs:\\"compressible\\" AND abs:\\"surrogate\\""),
    ("Q2","abs:\\"model discrepancy\\" AND abs:\\"convection\\""),
    ("Q3","abs:\\"radiation\\" AND abs:\\"natural convection\\" AND abs:\\"neural network\\""),
    ("Q3","abs:\\"radiative\\" AND abs:\\"buoyant\\" AND abs:\\"learning\\""),
    ("Q4","abs:\\"geometric scaling\\" AND abs:\\"neural network\\""),
    ("Q4","abs:\\"scale generalization\\" AND abs:\\"neural operator\\""),
    ("Q4","abs:\\"trained on small\\" AND abs:\\"larger\\""),
    ("Q4","abs:\\"length scale\\" AND abs:\\"generalization\\" AND abs:\\"surrogate\\""),
    ("Q5","abs:\\"discrepancy\\" AND abs:\\"multi-fidelity\\" AND abs:\\"heat transfer\\""),
    ("Q5","abs:\\"analytic\\" AND abs:\\"correction\\" AND abs:\\"surrogate\\" AND abs:\\"heat transfer\\""),
    ("Q5","abs:\\"hybrid\\" AND abs:\\"low-fidelity solver\\" AND abs:\\"neural\\""),
]
'''
s = s.replace('if __name__ == "__main__" and len(sys.argv) > 1:', extra + 'if __name__ == "__main__" and len(sys.argv) > 1:')
open(p,"w",encoding="utf-8").write(s)
print("ok")
