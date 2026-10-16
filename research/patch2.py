src=open("train_fno.py",encoding="utf-8").read()
src=src.replace('p.add_argument("--out",default="")','p.add_argument("--target",default="Y"); p.add_argument("--out",default="")')
src=src.replace('        x=Xtr_n[j].to(dev); y=Ytr_n[j].to(dev); pp=Ptr[perm[s:s+a.batch]].to(dev)',
                '        x=Xtr_n[j].to(dev); y=(Ytr_n[j]-Xtr_n[j]).to(dev) if a.target=="delta" else Ytr_n[j].to(dev); pp=Ptr[perm[s:s+a.batch]].to(dev)')
src=src.replace('pte=infer(Xte_n,Pte)','pte=infer(Xte_n,Pte)\nif a.target=="delta": pte=pte+Xte_n')
src=src.replace('res=dict(arm=a.arm,','res=dict(arm=a.arm,target=a.target,')
open("train_fno.py","w",encoding="utf-8").write(src); print("patched")
