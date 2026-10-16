"""统一口径消融（§5.4-5.6 重跑版）：5 个变体 x 2 种划分 x 3 个种子，报 mean±std。

  A  eps-blind          4 通道 -> 直接预测 D
  B  eps 作输入          5 通道 -> 直接预测 D
  P1 eps 进网络、输出端乘 eps              5 通道 -> P，D = eps*P
  P2 eps 进网络、输出端乘 (eps+b*eps^2)    同上，b 可学
  P3 eps 完全不进网络    4 通道 -> P，D = eps*P   <-- 「单温差数据集可复用」那一条
"""
import numpy as np, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, gerr, CH
from harness import Plain, Prior, run_one

OUT = os.path.expanduser('~/nc_run/p7'); os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
EPOCHS, SEEDS = 200, [0, 1, 2]
VARIANTS = [('A_eps_blind',       lambda: Plain(False)),
            ('B_eps_input',       lambda: Plain(True)),
            ('P1_eps_in_x_eps',   lambda: Prior(False, True)),
            ('P2_eps_in_x_quad',  lambda: Prior(True,  True)),
            ('P3_eps_not_in_net', lambda: Prior(False, False))]

X, Y, eps, meta = load600()
D = Y - X
E = torch.tensor(eps).view(-1, 1, 1, 1).repeat(1, 1, X.shape[2], X.shape[3])
res = {'epochs': EPOCHS, 'seeds': SEEDS, 'n': int(len(eps)), 'width': 32,
       'note': '归一化统计量只用训练子集；目标一律 D；U-Net width 32 200 epoch'}
JF = os.path.join(OUT, 'exp_ml_final.json')
save = lambda: json.dump(res, open(JF, 'w'), indent=1)

for split in ['extrap', 'interp']:
    tr, te = splits(eps, split)
    IN4, sx = build_input(X, eps, False, tr)
    fl = gerr(X[te], Y[te])
    res[split] = dict(floor=fl, n_train=int(tr.sum()), n_test=int(te.sum()),
                      sx=[float(v) for v in sx.flatten()])
    print('\n=== %s  n_train=%d n_test=%d ===' % (split, tr.sum(), te.sum()), flush=True)
    for tag, mk in VARIANTS:
        acc, betas, ts = [], [], []
        print('    %s' % tag, flush=True)
        for sd_ in SEEDS:
            g, b, t = run_one(mk, IN4, E, D, tr, te, X[te], Y[te], sd_, epochs=EPOCHS,
                              dev=DEV, tag=tag, verbose=True)
            acc.append(g); betas.append(b); ts.append(t)
            print('      >> %-18s seed=%d  %s  %.0fs' % (tag, sd_,
                  ' '.join('%s=%.4f' % (CH[i], g[i]) for i in range(4)), t), flush=True)
        a = np.array(acc)
        res[split][tag] = dict(mean=a.mean(0).tolist(), std=a.std(0).tolist(), raw=acc,
                               beta=betas, train_s=ts,
                               gain_vs_floor=(np.array(fl) / a.mean(0)).tolist())
        save()
print('\nsaved %s' % JF)