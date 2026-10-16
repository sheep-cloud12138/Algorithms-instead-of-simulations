import torch, torch.nn as nn, torch.nn.functional as F, math
class SpectralConv2d(nn.Module):
    def __init__(self, ci, co, m1, m2):
        super().__init__()
        self.m1,self.m2=m1,m2
        s=1.0/max(ci,1)
        self.w1=nn.Parameter(s*torch.rand(ci,co,m1,m2,dtype=torch.cfloat))
        self.w2=nn.Parameter(s*torch.rand(ci,co,m1,m2,dtype=torch.cfloat))
    def forward(self,x):
        B,C,H,W=x.shape
        x_ft=torch.fft.rfft2(x)
        m1=min(self.m1,H//2); m2=min(self.m2,W//2+1)
        out=torch.zeros(B,self.w1.shape[1],H,W//2+1,dtype=torch.cfloat,device=x.device)
        out[:,:,:m1,:m2]=torch.einsum("bixy,ioxy->boxy",x_ft[:,:,:m1,:m2],self.w1[:,:,:m1,:m2])
        out[:,:,-m1:,:m2]=torch.einsum("bixy,ioxy->boxy",x_ft[:,:,-m1:,:m2],self.w2[:,:,:m1,:m2])
        return torch.fft.irfft2(out,s=(H,W))
class FNO(nn.Module):
    def __init__(self,in_ch,out_ch=4,width=32,modes=12,depth=4):
        super().__init__()
        self.fc0=nn.Conv2d(in_ch,width,1)
        self.spec=nn.ModuleList([SpectralConv2d(width,width,modes,modes) for _ in range(depth)])
        self.w=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(depth)])
        self.fc1=nn.Conv2d(width,128,1); self.fc2=nn.Conv2d(128,out_ch,1)
    def forward(self,x):
        x=self.fc0(x)
        for s,w in zip(self.spec,self.w): x=F.gelu(s(x)+w(x))
        return self.fc2(F.gelu(self.fc1(x)))
if __name__=="__main__":
    m=FNO(4+6).cuda()
    x=torch.randn(4,10,64,64).cuda()
    y=m(x); print("out",y.shape,"params",sum(p.numel() for p in m.parameters())/1e6,"M")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(20): m(x)
    print("peak GPU MB", torch.cuda.max_memory_allocated()/1e6)
