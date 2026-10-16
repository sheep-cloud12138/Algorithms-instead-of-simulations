"""公开 5000 组 classical-cavity（train_legacy，eps≡1.2，等温侧壁）上训练「朴素」与「物理先验」代理，
再搬到我们自己的随机壁面数据上（含 eps 外推段）。

与 exp_public_transfer.py 的区别：那份用的是同族的随机壁面 4000 组，这份用的是**另一个边界条件族**
（经典方腔），所以它同时检验 eps 外推和 BC 族外推。结果与 4000 组那批并排列出。
"""
import numpy as np, json, os, sys, csv
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, train_net, predict_net, report, CH
from common import UNet

LD = "/mnt/d/Desktop/算法代替仿真/research/data/train_legacy"
OUT = os.path.expanduser('~/nc_run/p7'); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
JF, CF = os.path.join(OUT, 'public_legacy_transfer.json'), os.path.join(OUT, 'pub_legacy_nets.pt')
res = json.load(open(JF)) if os.path.exists(JF) else {}
cache = torch.load(CF, weights_only=False) if os.path.exists(CF) else {}
save = lambda: json.dump(res, open(JF, 'w'), indent=1)

rows = list(csv.DictReader(open(LD + '/per_sample.csv')))
N = len(rows)
dT = np.array([float(r['dT']) for r in rows], np.float32)
T0 = np.array([float(r['T0']) for r in rows], np.float32)
eps = dT / T0
Ra = np.array([float(r['Ra']) for r in rows])
Xr = torch.load(LD + '/X.pt', weights_only=False, mmap=True).float()
Yr = torch.load(LD + '/Y.pt', weights_only=False, mmap=True).float()
X = Xr.clone(); Y = Yr.clone(); del Xr, Yr
for k in range(N):
    X[k, 2] = (X[k, 2] - (1 - dT[k] / (2 * T0[k]))) / eps[k]
    Y[k, 2] = (Y[k, 2] - (1 - dT[k] / (2 * T0[k]))) / eps[k]
print('公开 classical-cavity 训练集 %d 组, Ra %.2e-%.2e, eps≡%.2f' % (N, Ra.min(), Ra.max(), float(np.median(eps))), flush=True)
D = Y - X
IN, _ = build_input(X, np.full(N, float(np.median(eps)), np.float32), False)
tr = np.ones(N, bool)
nets = {}
for key, T in [('legacy_naive', D), ('legacy_prior', D / float(np.median(eps)))]:
    if key in cache:
        net = UNet(4, 32); net.load_state_dict(cache[key]['sd'])
        print('      [cache] %s' % key, flush=True)
        nets[key] = (net, cache[key]['sD'], cache[key].get('ts', 0.0)); continue
    net, sD, ts = train_net(UNet(4, 32), IN, T, tr, epochs=150, seed=0, DEV=DEV, bs=32, tag=key)
    cache[key] = dict(sd=net.state_dict(), sD=sD, ts=ts); torch.save(cache, CF)
    nets[key] = (net, sD, ts)
del IN, D, X, Y

X6, Y6, eps6, _ = load600(); D6 = Y6 - X6
ev6 = torch.tensor(eps6).view(-1, 1, 1, 1)
for split in ['extrap', 'interp']:
    tr6, te = splits(eps6, split)
    IN6, _ = build_input(X6, eps6, False)
    fl = report('%s floor' % split, X6[te], Y6[te], X6[te])
    d = res.setdefault(split, dict(floor=fl['g'], n_test=int(te.sum())))
    for key, (net, sD, _) in nets.items():
        Tp = predict_net(net, IN6, None, sD, te, DEV)
        Yh = X6[te] + ev6[te] * Tp if 'prior' in key else X6[te] + Tp
        g = report('公开5000(经典方腔) -> %s %s' % (split, key), Yh, Y6[te], X6[te])
        d[key] = dict(g=g['g'], gain=g['gain']); save()
# 对照：本机训练集同预算的先验（与另一份脚本共用缓存文件，若已存在就直接复用）
PC = os.path.join(OUT, 'pub_nets.pt')
pcache = torch.load(PC, weights_only=False) if os.path.exists(PC) else {}
for split in ['extrap', 'interp']:
    tr6, te = splits(eps6, split)
    IN6, _ = build_input(X6, eps6, False)
    key = 'own_prior_%s' % split
    if key in pcache:
        net = UNet(4, 32); net.load_state_dict(pcache[key]['sd']); sD = pcache[key]['sD']
    else:
        net, sD, _ = train_net(UNet(4, 32), IN6, D6 / ev6, tr6, epochs=150, seed=0, DEV=DEV, bs=32, tag=key)
        pcache[key] = dict(sd=net.state_dict(), sD=sD); torch.save(pcache, PC)
    Tp = predict_net(net, IN6, None, sD, te, DEV)
    g = report('%s own_prior (n=%d)' % (split, tr6.sum()), X6[te] + ev6[te] * Tp, Y6[te], X6[te])
    res[split]['own_prior'] = dict(g=g['g'], gain=g['gain'], n_train=int(tr6.sum())); save()
save()
print('\nsaved %s' % JF)