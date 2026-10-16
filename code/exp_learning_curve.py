"""学习曲线：训练样本量 vs 测试误差。对比「有物理先验 D=eps*P(X)」与「无先验」。
测试集固定（interp 划分的 200 组），训练集从剩余 400 组里抽 n 个，3 个随机种子。"""
import numpy as np, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, train_net, predict_net, report, CH
from common import UNet

OUT = os.path.expanduser('~/nc_run/p7'); os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)

X, Y, eps, meta = load600()
tr, te = splits(eps, 'interp')
pool = np.where(tr)[0]
print('测试集 %d 组，训练池 %d 组' % (te.sum(), len(pool)), flush=True)
ev = torch.tensor(eps).view(-1,1,1,1)
NS = [50, 100, 200, 400]; SEEDS = [0,1,2]
curves = {'B_eps_input': {}, 'C_prior': {}}
Yt = Y[te]; fl = report('floor(do nothing)', X[te], Yt, X[te])

for n in NS:
    for var in ['B_eps_input', 'C_prior']:
        acc = []
        for sd in SEEDS:
            rng = np.random.RandomState(1000+sd)
            sub = rng.choice(pool, size=min(n, len(pool)), replace=False)
            m = np.zeros(len(eps), bool); m[sub] = True
            use_eps = (var == 'B_eps_input')
            IN, _ = build_input(X, eps, use_eps)
            T = Y if var == 'B_eps_input' else X + (Y-X)/ev
            net, sD, ts = train_net(UNet(5 if use_eps else 4, 32), IN, T, m, epochs=200,
                                    seed=sd, DEV=DEV, tag='%s n=%d s=%d' % (var, n, sd), log=False)
            Tp = predict_net(net, IN, T, sD, te, DEV)
            Yp = Tp if var == 'B_eps_input' else X[te] + (Tp - X[te])*ev[te]
            g = report('%s n=%3d seed=%d' % (var, n, sd), Yp, Yt, X[te], extra='%.0fs' % ts)
            acc.append(g['g'])
        a = np.array(acc)
        curves[var][n] = dict(mean=a.mean(0).tolist(), std=a.std(0).tolist(), raw=a.tolist())
        print('   === %s  n=%3d  平均 ux=%.4f uy=%.4f th=%.4f p=%.4f' %
              (var, n, *a.mean(0)), flush=True)
        json.dump(dict(curves=curves, floor=fl['g'], n_test=int(te.sum()), n_pool=int(len(pool)),
                       seeds=SEEDS), open(os.path.join(OUT,'learning_curve.json'),'w'), indent=1)
print('saved ~/nc_run/p7/learning_curve.json')