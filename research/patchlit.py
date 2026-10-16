import re, io, os
p = os.path.join("research","lit_review.py")
s = open(p, encoding="utf-8").read()
s = s.replace('run(sys.argv[1], B[sys.argv[1]])',
              'if __name__ == "__main__" and len(sys.argv) > 1:\n    run(sys.argv[1], B[sys.argv[1]])')
extra = '''\nB["6"] = [
    ("Q1","abs:\\"physical units\\" AND abs:\\"neural operator\\""),
    ("Q1","abs:\\"scale equivariance\\" AND abs:\\"PDE\\""),
    ("Q1","abs:\\"unit-aware\\" AND abs:\\"neural network\\""),
    ("Q1","abs:\\"dimensionless numbers\\" AND abs:\\"deep learning\\""),
    ("Q1","ti:\\"dimensional\\" AND abs:\\"operator learning\\""),
    ("Q1","abs:\\"similarity\\" AND abs:\\"dimensionless\\" AND abs:\\"neural network\\""),
    ("Q6","abs:\\"exact symmetry\\" AND abs:\\"machine learning\\" AND abs:\\"physics\\""),
    ("Q6","abs:\\"group equivariant\\" AND abs:\\"neural operator\\""),
    ("Q6","abs:\\"Wigner\\" AND abs:\\"dimensional analysis\\" AND abs:\\"machine learning\\""),
    ("Q6","abs:\\"quantify\\" AND abs:\\"similarity solution\\" AND abs:\\"deep learning\\""),
    ("Q6","abs:\\"self-similar\\" AND abs:\\"neural network\\" AND abs:\\"convection\\""),
    ("Q6","abs:\\"thermal radiation\\" AND abs:\\"natural convection\\" AND abs:\\"CFD\\" AND abs:\\"cavity\\""),
]
'''
s = s.replace('if __name__ == "__main__" and len(sys.argv) > 1:', extra + 'if __name__ == "__main__" and len(sys.argv) > 1:')
open(p,"w",encoding="utf-8").write(s)
print("patched")
