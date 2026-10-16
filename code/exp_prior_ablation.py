"""先验形式消融：D = eps*P(X)  还是  D = (eps + beta*eps^2)*P(X)。
net 从第 5 通道读 eps，只预测 P；输出端乘回 eps 或 eps+beta*eps^2（beta 可学习）。
对比 B = 普通 U-Net（输入带 eps，直接预测 D）。"""
import numpy as np, json, os, sys, time
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, train_net, predict_net, report, CH
from common import UNet

OUT = os.path.expanduser('~/nc_run/p7'); os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)

class Scaled(nn.Module):
    """输入 x = 4 通道 X + 第 5 通道 eps；输出 D = s(eps) * P(X)。"""
    def __init__(self, mode, width=32):
        super().__init__()
        self.base = UNet(5, width); self.mode = mode
        self.beta = nn.Parameter(torch.zeros(4))
    def forward(self, x):
        e = x[:, 4:5]; p = self.base(x)
        if self.mode == 'eps':
            return e * p
        return (e + self.beta.view(1, 4, 1, 1) * e**2) * p

X, Y, eps, meta = load600()
ev = torch.tensor(eps).view(-1, 1, 1, 1); D = Y - X
IN, _ = build_input(X, eps, True)          # 5 通道：X 加 eps 常数场
res = {}
for split in ['extrap', 'interp']:
    tr, te = splits(eps, split)
    Yt = Y[te]; fl = report('%s floor' % split, X[te], Yt, X[te])
    runs = {'B_plain_no_prior': UNet(5, 32), 'P1_scaled_eps': Scaled('eps'), 'P2_scaled_eps_quad': Scaled('quad')}
    res[split] = dict(floor=fl['g'], n_test=int(te.sum()), n_train=int(tr.sum()))
    for tag, net in runs.items():
        net, sD, ts = train_net(net, IN, D, tr, epochs=200, seed=0, DEV=DEV, tag=tag)
        Tp = predict_net(net, IN, D, sD, te, DEV)
        Yp = X[te] + Tp
        g = report('%s %s' % (split, tag), Yp, Yt, X[te], extra='%.0fs' % ts)
        beta = None
        for m in net.modules():
            if hasattr(m, 'beta'):
                beta = [float(v) for v in m.beta.detach().cpu().flatten()]; break
        res[split][tag] = dict(g=g['g'], gain=g['gain'], beta=beta, train_s=ts)
        if beta: print('        beta =', ['%.3f' % v for v in beta], flush=True)
        json.dump(res, open(os.path.join(OUT, 'prior_ablation.json'), 'w'), indent=1)
print('saved ~/nc_run/p7/prior_ablation.json')