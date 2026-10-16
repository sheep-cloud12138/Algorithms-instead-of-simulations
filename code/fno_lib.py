"""FNO2d（谱算子学习）骨干，接口与 harness 约定一致：forward(x, e) -> 预测 D。"""
import torch, torch.nn as nn, torch.nn.functional as F

class SConv(nn.Module):
    def __init__(self, ci, co, m):
        super().__init__(); self.m = m
        self.w = nn.Parameter(torch.randn(ci, co, m, m, 2) / (ci * co))
    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x, norm='ortho')
        m1 = min(self.m, H // 2); m2 = min(self.m, W // 2 + 1)
        out = torch.zeros(B, self.w.shape[1], H, W // 2 + 1, dtype=torch.complex64, device=x.device)
        w = torch.view_as_complex(self.w[:, :, :m1, :m2].contiguous())
        out[:, :, :m1, :m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :m1, :m2], w)
        return torch.fft.irfft2(out, s=(H, W), norm='ortho')

class FNO2d(nn.Module):
    def __init__(self, ci, width=32, modes=12, layers=4):
        super().__init__()
        self.fc0 = nn.Conv2d(ci, width, 1)
        self.sc = nn.ModuleList([SConv(width, width, modes) for _ in range(layers)])
        self.lw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Conv2d(width, 128, 1); self.fc2 = nn.Conv2d(128, 4, 1)
    def forward(self, x):
        x = self.fc0(x)
        for s, l in zip(self.sc, self.lw):
            x = F.gelu(s(x) + l(x))
        return self.fc2(F.gelu(self.fc1(x)))

class FNOPlain(nn.Module):
    def __init__(self, eps_in, **kw):
        super().__init__(); self.eps_in = eps_in; self.base = FNO2d(5 if eps_in else 4, **kw)
    def forward(self, x, e):
        return self.base(torch.cat([x, e], 1) if self.eps_in else x)

class FNOPrior(nn.Module):
    def __init__(self, quad, eps_in, **kw):
        super().__init__(); self.quad, self.eps_in = quad, eps_in
        self.base = FNO2d(5 if eps_in else 4, **kw); self.beta = nn.Parameter(torch.zeros(4))
    def forward(self, x, e):
        p = self.base(torch.cat([x, e], 1) if self.eps_in else x)
        s = e + self.beta.view(1, 4, 1, 1) * e ** 2 if self.quad else e
        return s * p