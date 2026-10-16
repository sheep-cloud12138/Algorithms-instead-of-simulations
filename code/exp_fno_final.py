"""强基线复跑：FNO2d x {A 无 eps, B eps 作输入, P3 eps 不进网络但输出端乘 eps}，3 个种子。

与 exp_ml_final.py 完全同一套划分、归一化（只用训练子集）、评价指标，只是把骨干换成 FNO。
"""
import numpy as np, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, gerr, CH
from harness import run_one
from fno_lib import FNOPlain, FNOPrior, FNO2d

OUT = os.path.expanduser('~/nc_run/p7'); os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
EPOCHS, SEEDS = 200, [0, 1, 2]
VARIANTS = [('A_eps_blind',       lambda: FNOPlain(False)),
            ('B_eps_input',       lambda: FNOPlain(True)),
            ('P3_eps_not_in_net', lambda: FNOPrior(False, False))]

X, Y, eps, meta = load600()
D = Y - X
E = torch.tensor(eps).view(-1, 1, 1, 1).repeat(1, 1, X.shape[2], X.shape[3])
npar = sum(p.numel() for p in FNO2d(5).parameters())
res = {'epochs': EPOCHS, 'seeds': SEEDS, 'n_params': npar,
       'note': 'FNO2d modes=12 width=32 layers=4；归一化统计量只用训练子集；目标一律 D'}
print('FNO2d 参数量 %.0fk' % (npar / 1e3), flush=True)
JF = os.path.join(OUT, 'exp_fno_final.json')
save = lambda: json.dump(res, open(JF, 'w'), indent=1)

for split in ['extrap', 'interp']:
    tr, te = splits(eps, split)
    IN4, sx = build_input(X, eps, False, tr)
    fl = gerr(X[te], Y[te])
    res[split] = dict(floor=fl, n_train=int(tr.sum()), n_test=int(te.sum()))
    print('\n=== %s  n_train=%d n_test=%d ===' % (split, tr.sum(), te.sum()), flush=True)
    for tag, mk in VARIANTS:
        acc, ts = [], []
        print('    %s' % tag, flush=True)
        for sd_ in SEEDS:
            g, b, t = run_one(mk, IN4, E, D, tr, te, X[te], Y[te], sd_, epochs=EPOCHS, dev=DEV, tag=tag)
            acc.append(g); ts.append(t)
            print('      >> %-18s seed=%d  %s  %.0fs' % (tag, sd_,
                  ' '.join('%s=%.4f' % (CH[i], g[i]) for i in range(4)), t), flush=True)
        a = np.array(acc)
        res[split][tag] = dict(mean=a.mean(0).tolist(), std=a.std(0).tolist(), raw=acc,
                               train_s=ts, gain_vs_floor=(np.array(fl) / a.mean(0)).tolist())
        save()
print('\nsaved %s' % JF)