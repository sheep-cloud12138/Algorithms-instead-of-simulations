import os
p="research/lit_review.py"; s=open(p,encoding="utf-8").read()
extra = '''\nB["8"] = [
    ("Q1","abs:\\"equivariant\\" AND abs:\\"units\\""),
    ("Q1","abs:\\"dimensional analysis\\" AND abs:\\"neural network\\" AND abs:\\"PDE\\""),
    ("Q1","abs:\\"nondimensionalization\\" AND abs:\\"neural\\""),
    ("Q1","abs:\\"physical units\\" AND abs:\\"physics-informed\\""),
    ("Q3","abs:\\"conjugate heat transfer\\" AND abs:\\"radiation\\" AND abs:\\"machine learning\\""),
    ("Q3","abs:\\"surface radiation\\" AND abs:\\"surrogate\\" AND abs:\\"heat\\""),
    ("Q3","abs:\\"natural convection\\" AND abs:\\"radiation\\" AND abs:\\"surrogate\\" AND abs:\\"enclosure\\""),
    ("Q6","abs:\\"similarity solution\\" AND abs:\\"neural network\\""),
    ("Q6","abs:\\"asymptotic\\" AND abs:\\"correction\\" AND abs:\\"neural network\\" AND abs:\\"convection\\""),
    ("Q6","abs:\\"self-similarity\\" AND abs:\\"deep learning\\" AND abs:\\"flow\\""),
    ("Q6","abs:\\"invariant\\" AND abs:\\"scaling\\" AND abs:\\"neural operator\\""),
]
'''
s = s.replace('if __name__ == "__main__" and len(sys.argv) > 1:', extra + 'if __name__ == "__main__" and len(sys.argv) > 1:')
open(p,"w",encoding="utf-8").write(s); print("ok")
