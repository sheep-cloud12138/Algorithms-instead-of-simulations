"""FiLM-conditioned U-Net baseline: same backbone, eps enters through per-layer
scale/shift instead of a constant input channel."""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, sd_of, CH
sys.path.insert(0, os.path.expanduser('~/nc_run/p4'))
from common import Blk

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class FiLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 2 * c))
        nn.init.zeros_(self.m[-1].weight); nn.init.zeros_(self.m[-1].bias)
    def forward(self, x, e):
        g, b = self.m(e).chunk(2, 1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]

class FiLMUNet(nn.Module):
    def __init__(self, ci=4, w=32):
        super().__init__()
        self.e1 = Blk(ci, w); self.e2 = Blk(w, 2 * w); self.e3 = Blk(2 * w, 4 * w)
        self.b = Blk(4 * w, 4 * w)
        self.d3 = Blk(4 * w + 4 * w, 2 * w); self.d2 = Blk(2 * w + 2 * w, w)
        self.d1 = Blk(w + w, w); self.o = nn.Conv2d(w, 4, 1)
        self.f1, self.f2, self.f3 = FiLM(w), FiLM(2 * w), FiLM(4 * w)
        self.fb, self.fd3, self.fd2, self.fd1 = FiLM(4 * w), FiLM(2 * w), FiLM(w), FiLM(w)
    def forward(self, x, e):
        x1 = self.f1(self.e1(x), e)
        x2 = self.f2(self.e2(F.max_pool2d(x1, 2)), e)
        x3 = self.f3(self.e3(F.max_pool2d(x2, 2)), e)
        b = self.fb(self.b(F.max_pool2d(x3, 2)), e)
        y = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        y = self.fd3(self.d3(torch.cat([y, x3], 1)), e)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        y = self.fd2(self.d2(torch.cat([y, x2], 1)), e)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        y = self.fd1(self.d1(torch.cat([y, x1], 1)), e)
        return self.o(y)

def train(net, IN, E, T, tr, epochs=200, seed=0, bs=16):
    torch.manual_seed(seed); np.random.seed(seed)
    sD = sd_of(T[tr], int(tr.sum()))
    Tn = T / sD
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx = np.where(tr)[0].copy(); bs = min(bs, max(2, len(idx) // 2)); t0 = time.time()
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            pr = net(IN[b].to(DEV), E[b].to(DEV)); dt = Tn[b].to(DEV)
            rel = ((pr - dt).pow(2).mean((1, 2, 3)) / (dt.pow(2).mean((1, 2, 3)) + 1e-12)).mean()
            opt.zero_grad(); rel.backward(); opt.step()
        sch.step()
    return net, sD, time.time() - t0

X, Y, eps, meta = load600()
D = Y - X
IN, _ = build_input(X, eps, False)
E = torch.tensor(eps.astype(np.float32)).view(-1, 1)
res = {}
for split in ['extrap', 'interp']:
    tr, te = splits(eps, split)
    for seed in [0, 1, 2]:
        net = FiLMUNet()
        net, sD, ts = train(net, IN, E, D, tr, seed=seed)
        net.eval()
        with torch.no_grad():
            pred = torch.zeros_like(D[te])
            for k in range(0, int(te.sum()), 32):
                ii = np.where(te)[0][k:k + 32]
                pred[k:k + 32] = net(IN[ii].to(DEV), E[ii].to(DEV)).cpu() * sD
        Yh = X[te] + pred
        g = [float(torch.norm(Yh[:, c] - Y[te, c]) / torch.norm(Y[te, c])) for c in range(4)]
        res.setdefault(split, []).append(dict(seed=seed, g=g, train_s=ts))
        print('%s seed %d  %s  %.0fs' % (split, seed, ' '.join('%.4f' % v for v in g), ts), flush=True)
out = {s: dict(mean=[float(np.mean([r['g'][c] for r in v])) for c in range(4)],
               std=[float(np.std([r['g'][c] for r in v])) for c in range(4)], runs=v)
       for s, v in res.items()}
json.dump(out, open('论文/film_baseline.json', 'w'), indent=1)
print(json.dumps({k: dict(mean=[round(x, 4) for x in v['mean']]) for k, v in out.items()}, indent=1))
