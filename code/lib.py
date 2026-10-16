"""公共库：数据加载、划分、U-Net/FNO 训练与评测（所有后续实验共用）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, os, sys, time

sys.path.insert(0, os.path.expanduser('~/nc_run/p4'))
from common import UNet                      # 4 通道输入 U-Net, width 可调
PACK = os.path.expanduser('~/nc_run/p2/pairs600_fixedp.npz')   # p 通道已按公开数据集配方修正
CH = ['ux','uy','theta','p']

def load600():
    z = np.load(PACK, allow_pickle=True)
    X = torch.tensor(z['X']); Y = torch.tensor(z['Y'])
    meta = json.loads(str(z['meta']))
    eps = np.array([m['eps'] for m in meta], np.float32)
    return X, Y, eps, meta

def splits(eps, kind):
    """extrap: 训 eps<=1.3 / 测 eps>1.3 ;  interp: 按 eps 分层每 3 个取 1 个做测试集"""
    if kind == 'extrap':
        return eps <= 1.3, eps > 1.3
    order = np.argsort(eps); te = np.zeros(len(eps), bool); te[order[::3]] = True
    return ~te, te

def build_input(X, eps, use_eps):
    N, _, R, _ = X.shape
    sx = X.reshape(N,4,-1).std(2).mean(0).view(1,4,1,1)
    IN = X / sx
    if use_eps:
        ec = torch.tensor(eps).view(-1,1,1,1).repeat(1,1,R,R)
        IN = torch.cat([IN, ec], 1)
    return IN, sx

def sd_of(T, N, keep=4):
    if keep == 0: return torch.ones(1,4,1,1)
    return T.reshape(N,4,-1)[:, :keep].std(2).mean(0).view(1,4,1,1)

def train_net(net, IN, T, tr, epochs=200, seed=0, DEV='cuda', bs=16, tag='', log=True, sD=None):
    torch.manual_seed(seed); np.random.seed(seed)
    N = IN.shape[0]
    if sD is None: sD = sd_of(T, N)
    Tn = T / sD
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx = np.where(tr)[0].copy(); t0 = time.time(); bs = min(bs, max(2, len(idx)//2))
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx); tot = 0.; nb = 0
        for k in range(0, len(idx), bs):
            b = idx[k:k+bs]
            pr = net(IN[b].to(DEV)); dt = Tn[b].to(DEV)
            rel = ((pr-dt).pow(2).mean((1,2,3)) / (dt.pow(2).mean((1,2,3))+1e-12)).mean()
            opt.zero_grad(); rel.backward(); opt.step(); tot += float(rel.detach()); nb += 1
        sch.step()
        if log and ((ep+1) % 50 == 0):
            print('      %-22s ep %4d/%d relMSE %.5f %5.0fs' % (tag, ep+1, epochs, tot/nb, time.time()-t0), flush=True)
    return net, sD, time.time()-t0

def predict_net(net, IN, T, sD, te, DEV='cuda', bs=64):
    net.eval(); out = []
    idx = np.where(te)[0]
    with torch.no_grad():
        for k in range(0, len(idx), bs):
            out.append((net(IN[idx[k:k+bs]].to(DEV)).cpu() * sD))
    return torch.cat(out)

def gerr(Yp, Yt):
    return [float((((Yp[:,c]-Yt[:,c])**2).sum() / (Yt[:,c]**2).sum())**0.5) for c in range(4)]

def report(tag, Yp, Yt, Yfloor, extra=None):
    g = gerr(Yp, Yt); fl = gerr(Yfloor, Yt)
    line = '  >>> %-26s ' % tag + ' '.join('%s=%.4f' % (CH[i], g[i]) for i in range(4))
    line += '  [floor ' + ' '.join('%.4f' % v for v in fl) + ']'
    if extra: line += ' ' + extra
    print(line, flush=True)
    return dict(g=g, floor=fl, gain=[fl[i]/g[i] for i in range(4)])