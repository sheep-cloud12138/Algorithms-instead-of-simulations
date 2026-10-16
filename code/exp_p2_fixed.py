"""修正 p 通道后的主表：U-Net A（无 eps 输入）/ B（有 eps 输入），两个划分。"""
import numpy as np, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, train_net, predict_net, report, CH
from common import UNet
OUT = os.path.expanduser('~/nc_run/p7'); os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
X, Y, eps, meta = load600()
res = {'pack': 'pairs600_fixedp.npz'}
for split in ['extrap', 'interp']:
    tr, te = splits(eps, split); Yt = Y[te]
    fl = report('%s floor' % split, X[te], Yt, X[te])
    res[split] = dict(floor=fl['g'], gain_floor_to_X=None, n_train=int(tr.sum()), n_test=int(te.sum()))
    for tag, use_eps in [('A_no_eps', False), ('B_eps', True)]:
        IN, _ = build_input(X, eps, use_eps)
        net, sD, ts = train_net(UNet(5 if use_eps else 4, 32), IN, Y-X, tr, epochs=200, seed=0, DEV=DEV, tag=tag)
        Yp = X[te] + predict_net(net, IN, Y-X, sD, te, DEV)
        g = report('%s %s' % (split, tag), Yp, Yt, X[te], extra='%.0fs' % ts)
        res[split][tag] = dict(g=g['g'], gain=g['gain'], train_s=ts)
        json.dump(res, open(os.path.join(OUT, 'p2_fixed.json'), 'w'), indent=1)
print('saved ~/nc_run/p7/p2_fixed.json')