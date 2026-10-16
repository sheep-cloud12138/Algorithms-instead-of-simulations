import re,io
src=open("train_fno.py",encoding="utf-8").read()
src=src.replace(
"""def relL2(pred,tgt):
    d=((pred-tgt)**2).sum(dim=(1,2,3)); n=(tgt**2).sum(dim=(1,2,3)).clamp_min(1e-12)
    return (torch.sqrt(d/n)).mean().item()""",
"""def relL2(pred,tgt):
    ax=tuple(range(1,pred.dim()))
    d=((pred-tgt)**2).sum(dim=ax); n=(tgt**2).sum(dim=ax).clamp_min(1e-12)
    return (torch.sqrt(d/n)).mean().item()""")
open("train_fno.py","w",encoding="utf-8").write(src)
print("patched")
