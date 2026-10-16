src=open("train_fno.py",encoding="utf-8").read()
src=src.replace('p.add_argument("--target",default="Y")','p.add_argument("--target",default="Y"); p.add_argument("--loss",default="target")')
src=src.replace('        loss=F.mse_loss(model(x),y)',
'''        pred=model(x)
        if a.loss=="recon" and a.target=="delta": pred=pred+x
        loss=F.mse_loss(pred,y)''')
src=src.replace('res=dict(arm=a.arm,target=a.target,','res=dict(arm=a.arm,target=a.target,loss=a.loss,')
open("train_fno.py","w",encoding="utf-8").write(src); print("patched")
