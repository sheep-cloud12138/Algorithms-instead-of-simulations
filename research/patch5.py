src=open("train_fno.py",encoding="utf-8").read()
old='y=(Ytr_n[j]-Xtr_n[j]).to(dev) if a.target=="delta" else Ytr_n[j].to(dev)'
new='y=(Ytr_n[j]-Xtr_n[j]).to(dev) if a.target=="delta" and a.loss=="target" else Ytr_n[j].to(dev)'
assert old in src
src=src.replace(old,new)
open("train_fno.py","w",encoding="utf-8").write(src); print("patched")
