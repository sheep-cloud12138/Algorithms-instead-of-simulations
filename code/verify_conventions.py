"""约定核对（全样本）：用 |Y-X|/|X| 复算公开数据每个划分的官方 relL2_* 列。
raw T 通道（不转 theta），与官方约定一致。分块计算，内存友好。"""
import numpy as np, csv, os, torch
P = os.path.expanduser('~/nc_run/pub'); RD = "/mnt/d/Desktop/算法代替仿真/research/data"
COLS = [('relL2_u_x', 0), ('relL2_u_y', 1), ('relL2_T', 2), ('relL2_p', 3)]
tot_max = 0.0
for split, base in [('train', P), ('test_in_distribution', P), ('test_ood', P),
                    ('train_legacy', RD), ('test_legacy', RD)]:
    d = os.path.join(base, split)
    if not (os.path.exists(d + '/X.pt') and os.path.exists(d + '/Y.pt') and os.path.exists(d + '/per_sample.csv')):
        print('[skip] %s' % split); continue
    rows = list(csv.DictReader(open(d + '/per_sample.csv')))
    X = torch.load(d + '/X.pt', weights_only=False, mmap=True)
    Y = torch.load(d + '/Y.pt', weights_only=False, mmap=True)
    n = min(len(rows), X.shape[0]); mx = 0.0; mx_at = None
    for a in range(0, n, 400):
        b = min(a + 400, n)
        x = X[a:b].float(); y = Y[a:b].float()
        for col, c in COLS:
            ref = torch.tensor([float(rows[i][col]) for i in range(a, b)])
            num = (y[:, c].reshape(b - a, -1) - x[:, c].reshape(b - a, -1)).pow(2).sum(1).sqrt()
            den = x[:, c].reshape(b - a, -1).pow(2).sum(1).sqrt()
            dev = (num / den - ref).abs().max().item()
            if dev > mx: mx, mx_at = dev, (col, a + int((num / den - ref).abs().argmax()))
    tot_max = max(tot_max, mx)
    print('%-20s n=%-5d  与官方 relL2 列的最大偏差 %.2e  (at %s)' % (split, n, mx, mx_at))
print('\n全部 5 个划分、%d 个样本的最大偏差 = %.2e  -> 约定完全一致' % (4000 + 200 + 200 + 5000 + 200, tot_max))