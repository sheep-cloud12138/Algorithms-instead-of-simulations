src=open("train_fno.py",encoding="utf-8").read()
old='        x=Xtr_n[j].to(dev); y=(Ytr_n[j]-Xtr_n[j]).to(dev) if a.target=="delta" else Ytr_n[j].to(dev); pp=Ptr[perm[s:s+a.batch]].to(dev)\n        if pp.shape[1]>0: x=torch.cat([x,pp[:,:,None,None].expand(-1,-1,64,64)],1)'
new='        xf=Xtr_n[j].to(dev); x=xf; y=(Ytr_n[j]-Xtr_n[j]).to(dev) if a.target=="delta" else Ytr_n[j].to(dev); pp=Ptr[perm[s:s+a.batch]].to(dev)\n        if pp.shape[1]>0: x=torch.cat([x,pp[:,:,None,None].expand(-1,-1,64,64)],1)'
assert old in src
src=src.replace(old,new)
src=src.replace('if a.loss=="recon" and a.target=="delta": pred=pred+x','if a.loss=="recon" and a.target=="delta": pred=pred+xf')
open("train_fno.py","w",encoding="utf-8").write(src); print("patched")
