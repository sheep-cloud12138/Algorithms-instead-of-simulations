"""公开数据集（Menglik et al. 2026, 4000 组, 单一温差 dT=1.2）上的外部验证 + 跨数据集物理先验迁移。

(1) 约定核对 + 公开数据 eps 切片上的破缺律（R=|Y-X|/|Y|），并检验律是否与 Ra 无关
(2) 在公开 4000 组上训练「朴素」与「物理先验 D=eps*P(X)」两个代理，直接搬到我们自己的
    600 组上（含 eps 外推段），检验「单温差数据集能否被复用」这一核心卖点
(3) 对照：同等训练预算、只用我们自己训练集的先验模型

模型权重缓存到 pub_nets.pt，重跑脚本不会重训。结果分阶段增量写入 public_transfer.json。
"""
import numpy as np, json, os, sys, csv
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, train_net, predict_net, report, CH
from common import UNet

PUB = os.path.expanduser('~/nc_run/pub'); OUT = os.path.expanduser('~/nc_run/p7')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
DTP = 1.2                                   # 公开数据唯一温差；θ =(T-(1-DTP/2))/DTP
JF, CF = os.path.join(OUT, 'public_transfer.json'), os.path.join(OUT, 'pub_nets.pt')
res = json.load(open(JF)) if os.path.exists(JF) else {}
cache = torch.load(CF, weights_only=False) if os.path.exists(CF) else {}

def save():
    json.dump(res, open(JF, 'w'), indent=1)

def to_theta(T):
    return (T - (1.0 - DTP / 2.0)) / DTP

def train_cached(key, net, IN, T, tr, epochs=150):
    """带缓存的训练：同名 key 已训过就直接取权重。"""
    if key in cache:
        net.load_state_dict(cache[key]['sd']); net = net.to(DEV)   # 缓存里的权重在 CPU，必须搬回 DEV
        print('      [cache] %s 直接复用（训练时 %.0fs）' % (key, cache[key].get('ts', -1)), flush=True)
        return net, cache[key]['sD'], cache[key].get('ts', 0.0)
    net, sD, ts = train_net(net, IN, T, tr, epochs=epochs, seed=0, DEV=DEV, bs=32, tag=key)
    cache[key] = dict(sd=net.state_dict(), sD=sD, ts=ts)
    torch.save(cache, CF)
    print('    %s 训练 %.0fs（已缓存）' % (key, ts), flush=True)
    return net, sD, ts

# ================= 读公开数据 =================
Xp = torch.load(PUB + '/train/X.pt').float(); Yp = torch.load(PUB + '/train/Y.pt').float()
rows = list(csv.DictReader(open(PUB + '/train/per_sample.csv')))
Ra_p = np.array([float(r['Ra']) for r in rows])
Np = Xp.shape[0]
print('公开训练集 %d 组, Ra %.2e - %.2e, dT=%.1f' % (Np, Ra_p.min(), Ra_p.max(), DTP), flush=True)

# ---- 0. 约定核对（复算必须逐位等于官方 CSV，否则后面所有数字都不可信） ----
x0 = Xp[0].numpy(); y0 = Yp[0].numpy()
relX = [float(np.linalg.norm(x0[c] - y0[c]) / np.linalg.norm(x0[c])) for c in range(4)]
csvv = [float(rows[0]['relL2_u_x']), float(rows[0]['relL2_u_y']),
        float(rows[0]['relL2_T']), float(rows[0]['relL2_p'])]
print('约定核对  |Y-X|/|X| 复算 %s' % ['%.5f' % v for v in relX])
print('          官方 CSV        %s' % ['%.5f' % v for v in csvv], flush=True)
print('压力通道幅值比 |X_p|/|Y_p| = %.3f（本机修正后 0.93）'
      % float(np.median(np.linalg.norm(Xp[:, 3].reshape(Np, -1), axis=1)
                        / np.linalg.norm(Yp[:, 3].reshape(Np, -1), axis=1))), flush=True)
res['conv_check'] = dict(recomputed=relX, official=csvv, matched=bool(np.allclose(relX, csvv, atol=5e-6)))

# ---- 1. 公开数据的破缺幅值，与本机同 eps 组对照 ----
Xp3, Yp3 = Xp.clone(), Yp.clone()
Xp3[:, 2], Yp3[:, 2] = to_theta(Xp[:, 2]), to_theta(Yp[:, 2])
Dp = Yp3 - Xp3
Rp = np.stack([np.linalg.norm(Dp[:, c].reshape(Np, -1), axis=1)
               / np.linalg.norm(Yp3[:, c].reshape(Np, -1), axis=1) for c in range(4)], 1)
X6, Y6, eps6, _ = load600()
m12 = (eps6 >= 1.1) & (eps6 <= 1.3); D6 = Y6 - X6; ev6 = torch.tensor(eps6).view(-1, 1, 1, 1)
R6 = np.stack([np.linalg.norm(D6[:, c].reshape(len(eps6), -1), axis=1)
               / np.linalg.norm(Y6[:, c].reshape(len(eps6), -1), axis=1) for c in range(4)], 1)
print('\n=== 公开数据 eps=1.2 切片：R=|Y-X|/|Y| 中位数（本机同 eps 组对照）===')
print('%-8s %-32s %-30s' % ('通道', '公开 4000 组 (ε=1.2)', '本机 ε∈[1.1,1.3] n=%d' % m12.sum()))
for c in range(4):
    print('%-8s 中位 %.4f (p10 %.4f p90 %.4f)    中位 %.4f (p10 %.4f p90 %.4f)' %
          (CH[c], np.median(Rp[:, c]), np.percentile(Rp[:, c], 10), np.percentile(Rp[:, c], 90),
           np.median(R6[m12, c]), np.percentile(R6[m12, c], 10), np.percentile(R6[m12, c], 90)), flush=True)
from scipy import stats
print('\n=== 律是否与 Ra 无关（公开 4000 组，R 对 log10Ra 回归）===')
res['external_law'] = dict(n_pub=int(Np), n_own=int(m12.sum()), eps_pub=DTP,
                           R_pub_median=[float(np.median(Rp[:, c])) for c in range(4)],
                           R_pub_p10=[float(np.percentile(Rp[:, c], 10)) for c in range(4)],
                           R_pub_p90=[float(np.percentile(Rp[:, c], 90)) for c in range(4)],
                           R_own_median=[float(np.median(R6[m12, c])) for c in range(4)])
for c in range(4):
    s, i0, rv, pv, se = stats.linregress(np.log10(Ra_p), Rp[:, c])
    print('  %-6s slope=%+.4f  r=%+.3f  p=%.1e' % (CH[c], s, rv, pv), flush=True)
    res['external_law']['Ra_trend_%s' % CH[c]] = dict(slope=float(s), r=float(rv), p=float(pv))
save()

# ---- 2. 跨数据集先验迁移：公开 4000 组训练 -> 本机 600 组测试 ----
INp, _ = build_input(Xp3, np.full(Np, DTP, np.float32), False)   # 只喂 X，4 通道
tr_p = np.ones(Np, bool)
print('\n=== 在公开 4000 组上训练，然后搬到我们自己的数据上 ===', flush=True)
nets = {}
for key, T in [('pub_naive', Dp), ('pub_prior', Dp / DTP)]:
    nets[key] = train_cached(key, UNet(4, 32), INp, T, tr_p, epochs=150)
print('\n=== 迁移到本机 600 组 ===')
for split in ['extrap', 'interp']:
    tr, te = splits(eps6, split)
    IN6, _ = build_input(X6, eps6, False)
    fl = report('%s floor' % split, X6[te], Y6[te], X6[te])
    d = res.setdefault(split, dict(floor=fl['g'], n_test=int(te.sum())))
    for key, (net, sD, _) in nets.items():
        Tp = predict_net(net, IN6, None, sD, te, DEV)
        Yh = X6[te] + ev6[te] * Tp if 'prior' in key else X6[te] + Tp
        g = report('公开训练 -> %s %s' % (split, key), Yh, Y6[te], X6[te])
        d[key] = dict(g=g['g'], gain=g['gain'])
        save()
    # 对照：同样的网、同样的预算，但只用本机训练集
    net, sD, _ = train_cached('own_prior_%s' % split, UNet(4, 32), IN6, D6 / ev6, tr, epochs=150)
    Tp = predict_net(net, IN6, D6 / ev6, sD, te, DEV)
    g = report('%s own_prior (n=%d)' % (split, tr.sum()), X6[te] + ev6[te] * Tp, Y6[te], X6[te])
    d['own_prior'] = dict(g=g['g'], gain=g['gain'], n_train=int(tr.sum()))
    save()

save()
print('\nsaved %s' % JF)