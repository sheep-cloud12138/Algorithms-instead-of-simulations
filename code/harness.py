"""统一训练/评测骨架：§5.4-5.6 的所有对照都走这一份代码，保证口径一致。

约定：网络 forward(x, e) -> 直接预测 D = Y - X（在训练集尺度 sD 下训练）。
  x : 4 通道归一化输入（统计量只用训练子集）
  e : eps 常数场 (N,1,1,1)
"""
import numpy as np, torch, torch.nn as nn, time
from lib import sd_of, gerr
from common import UNet

class Plain(nn.Module):
    """直接预测 D。eps_in=True 时把 eps 拼成第 5 通道。"""
    def __init__(self, eps_in, width=32):
        super().__init__(); self.eps_in = eps_in; self.base = UNet(5 if eps_in else 4, width)
    def forward(self, x, e):
        return self.base(torch.cat([x, e], 1) if self.eps_in else x)

class Prior(nn.Module):
    """骨干只输出 P(X)；D = s(eps)*P。eps_in=False 时 eps 完全不进入骨干。"""
    def __init__(self, quad, eps_in, width=32):
        super().__init__(); self.quad, self.eps_in = quad, eps_in
        self.base = UNet(5 if eps_in else 4, width); self.beta = nn.Parameter(torch.zeros(4))
    def forward(self, x, e):
        p = self.base(torch.cat([x, e], 1) if self.eps_in else x)
        s = e + self.beta.view(1, 4, 1, 1) * e ** 2 if self.quad else e
        return s * p

def betas_of(net):
    for m in net.modules():
        if hasattr(m, 'beta'): return [float(v) for v in m.beta.detach().cpu().flatten()]
    return None

def run_one(mk, x, e, D, tr, te, Xte, Yte, seed, epochs=200, dev='cuda', bs=16, tag='', verbose=True):
    """训练一个变体并返回 (逐通道 relL2, 学到的 beta, 训练墙钟秒)。"""
    torch.manual_seed(seed); np.random.seed(seed)
    sD = sd_of(D[tr], int(tr.sum())); Dn = D / sD
    net = mk().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx = np.where(tr)[0].copy(); t0 = time.time(); bs = min(bs, max(2, len(idx) // 2))
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            pr = net(x[b].to(dev), e[b].to(dev)); dt = Dn[b].to(dev)
            rel = ((pr - dt).pow(2).mean((1, 2, 3)) / (dt.pow(2).mean((1, 2, 3)) + 1e-12)).mean()
            opt.zero_grad(); rel.backward(); opt.step()
        sch.step()
        if verbose and (ep + 1) % 50 == 0:
            print('        %-18s ep %3d/%d %5.0fs' % (tag, ep + 1, epochs, time.time() - t0), flush=True)
    net.eval(); out = []; ii = np.where(te)[0]
    with torch.no_grad():
        for k in range(0, len(ii), 64):
            out.append(net(x[ii[k:k + 64]].to(dev), e[ii[k:k + 64]].to(dev)).cpu() * sD)
    return gerr(Xte + torch.cat(out), Yte), betas_of(net), time.time() - t0