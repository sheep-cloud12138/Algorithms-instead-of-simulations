"""强基线：FNO2d（谱算子学习），与 U-Net 完全同一套划分、同一套归一化和评价指标。
A 无 eps 输入；B 有 eps 输入；C 有物理先验（预测 D/eps，X-only 输入，输出端乘 eps）。"""
import numpy as np, json, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, train_net, predict_net, report, CH

OUT = os.path.expanduser('~/nc_run/p7'); os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)

class SConv(nn.Module):
    def __init__(self, ci, co, m):
        super().__init__(); self.m = m
        self.w = nn.Parameter(torch.randn(ci, co, m, m, 2) / (ci*co))
    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x, norm='ortho')
        m1 = min(self.m, H//2); m2 = min(self.m, W//2+1)
        out = torch.zeros(B, self.w.shape[1], H, W//2+1, dtype=torch.complex64, device=x.device)
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

X, Y, eps, meta = load600()
ev = torch.tensor(eps).view(-1,1,1,1); D = Y - X
npar = sum(p.numel() for p in FNO2d(5).parameters())
print('FNO2d 参数量 %.0fk' % (npar/1e3), flush=True)
res = {'n_params': npar}
for split in ['extrap', 'interp']:
    tr, te = splits(eps, split)
    Yt = Y[te]; fl = report('%s floor' % split, X[te], Yt, X[te])
    res[split] = dict(floor=fl['g'], n_train=int(tr.sum()), n_test=int(te.sum()))
    for var in ['A_no_eps', 'B_eps', 'C_prior']:
        use_eps = (var == 'B_eps')
        IN, _ = build_input(X, eps, use_eps)
        T = X + D/ev if var == 'C_prior' else D
        net, sD, ts = train_net(FNO2d(5 if use_eps else 4), IN, T, tr, epochs=200, seed=0, DEV=DEV, tag=var)
        Tp = predict_net(net, IN, T, sD, te, DEV)
        Yp = X[te] + (Tp - X[te])*ev[te] if var == 'C_prior' else X[te] + Tp
        g = report('%s %s' % (split, var), Yp, Yt, X[te], extra='%.0fs' % ts)
        res[split][var] = dict(g=g['g'], gain=g['gain'], train_s=ts)
        json.dump(res, open(os.path.join(OUT,'fno_results.json'),'w'), indent=1)
print('saved ~/nc_run/p7/fno_results.json')