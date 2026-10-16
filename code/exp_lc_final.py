"""学习曲线复跑：B（eps 作输入）与 P3（eps 不进网络、输出端乘 eps），内插测试集 200 组，
n = 50/100/200/400，3 个种子。每个 n 的归一化统计量只用抽出来的那 n 个样本（不借测试集的光）。
"""
import numpy as np, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, gerr, CH
from harness import Plain, Prior, run_one

OUT = os.path.expanduser('~/nc_run/p7'); os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
EPOCHS, SEEDS, NS = 200, [0, 1, 2], [50, 100, 200, 400]
VARIANTS = [('B_eps_input', lambda: Plain(True)), ('P3_eps_not_in_net', lambda: Prior(False, False))]

X, Y, eps, meta = load600()
D = Y - X
E = torch.tensor(eps).view(-1, 1, 1, 1).repeat(1, 1, X.shape[2], X.shape[3])
tr, te = splits(eps, 'interp'); pool = np.where(tr)[0]
fl = gerr(X[te], Y[te])
res = {'epochs': EPOCHS, 'seeds': SEEDS, 'floor': fl, 'n_test': int(te.sum()),
       'n_pool': int(len(pool)), 'note': '归一化统计量只用抽中的 n 个样本'}
JF = os.path.join(OUT, 'exp_lc_final.json')
print('测试集 %d 组，训练池 %d 组' % (te.sum(), len(pool)), flush=True)
curves = {}
for n in NS:
    for tag, mk in VARIANTS:
        acc = []
        for sd_ in SEEDS:
            rng = np.random.RandomState(1000 + sd_)
            sub = rng.choice(pool, size=min(n, len(pool)), replace=False)
            m = np.zeros(len(eps), bool); m[sub] = True
            INn, _ = build_input(X, eps, False, m)      # 统计量只用抽中的样本
            g, b, t = run_one(mk, INn, E, D, m, te, X[te], Y[te], sd_, epochs=EPOCHS,
                              dev=DEV, tag='%s n=%d s=%d' % (tag, n, sd_), verbose=False)
            acc.append(g)
            print('   %-18s n=%3d seed=%d  %s  %.0fs' % (tag, n, sd_,
                  ' '.join('%s=%.4f' % (CH[i], g[i]) for i in range(4)), t), flush=True)
        a = np.array(acc)
        curves.setdefault(tag, {})[n] = dict(mean=a.mean(0).tolist(), std=a.std(0).tolist(), raw=acc)
        print('   === %s n=%3d 平均 %s' % (tag, n, ' '.join('%s=%.4f' % (CH[i], a.mean(0)[i]) for i in range(4))), flush=True)
        json.dump(dict(curves=curves, floor=fl, n_test=int(te.sum()), n_pool=int(len(pool)),
                       seeds=SEEDS, epochs=EPOCHS), open(JF, 'w'), indent=1)
print('\nsaved %s' % JF)