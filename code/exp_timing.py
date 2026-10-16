"""推理耗时 / 加速比实测。
求解器墙钟取自每个样本 meta.json 的 march.*.wall_s（本机真实耗时，非理论值）。"""
import numpy as np, json, os, glob, time, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load600, splits, build_input, sd_of, train_net, CH

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = os.path.expanduser('~/nc_run/p7'); os.makedirs(OUT, exist_ok=True)

# ---------- 1. 求解器墙钟 ----------
def walltimes(pat):
    w = []
    for m in glob.glob(pat):
        try: d = json.load(open(m))
        except Exception: continue
        a = d.get('march', {})
        t = 0.0; ok = True
        for k in ('bouss', 'comp'):
            if k in a: t += float(a[k].get('wall_s', np.nan))
        if np.isfinite(t) and t > 0: w.append((float(d['Ra']), float(d.get('dT', np.nan)), t))
    return np.array(w)

res = {}
for name, pat in [('batch600', os.path.expanduser('~/nc_run/batch/dataset/*/meta.json')),
                  ('paired100', os.path.expanduser('~/nc_run/paired/dataset/*/sample_000000/meta.json')),
                  ('eps0_27', os.path.expanduser('~/nc_run/eps0/dataset/*/meta.json'))]:
    W = walltimes(pat)
    if len(W) == 0: continue
    t = W[:, 2]
    res[name] = dict(n=len(t), median=float(np.median(t)), mean=float(t.mean()),
                     p10=float(np.percentile(t,10)), p90=float(np.percentile(t,90)), max=float(t.max()))
    print('%-10s n=%4d  中位 %7.1f s  均值 %7.1f s  p10 %6.1f  p90 %7.1f  最大 %7.1f' %
          (name, len(t), np.median(t), t.mean(), np.percentile(t,10), np.percentile(t,90), t.max()), flush=True)

# ---------- 2. 训练一个 U-Net 并计时推理 ----------
X, Y, eps, meta = load600()
tr, te = splits(eps, 'extrap')
IN, sxV = build_input(X, eps, True)
from common import UNet
torch.set_num_threads(16)
net, sD, tr_s = train_net(UNet(5, 32), IN, Y, tr, epochs=200, seed=0, DEV=DEV, tag='timing-net')
net = net.to('cpu').eval()

torch.set_num_threads(1)
with torch.no_grad():
    net(IN[:1]); t0 = time.time()
    for _ in range(20): net(IN[:1])
    cpu1 = (time.time()-t0)/20
torch.set_num_threads(24)
with torch.no_grad():
    net(IN[:1]); t0 = time.time()
    for _ in range(20): net(IN[:1])
    cpu24 = (time.time()-t0)/20
    t0 = time.time()
    for _ in range(5): net(IN[:256])
    cpu24b = (time.time()-t0)/5/256

net = net.to(DEV)
with torch.no_grad():
    net(IN[:1].to(DEV)); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): net(IN[:1].to(DEV))
    torch.cuda.synchronize(); gpu1 = (time.time()-t0)/50
    net(IN[:256].to(DEV)); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): net(IN[:256].to(DEV))
    torch.cuda.synchronize(); gpub = (time.time()-t0)/10/256
print('\n推理耗时（单样本）：CPU 1线程 %.1f ms | CPU 24线程 %.1f ms | GPU batch1 %.1f ms | GPU batch256 %.2f ms/样本'
      % (cpu1*1e3, cpu24*1e3, gpu1*1e3, gpub*1e3), flush=True)
res['inference'] = dict(cpu_1thread_s=cpu1, cpu_24thread_s=cpu24, cpu_batch256_s=cpu24b,
                        gpu_batch1_s=gpu1, gpu_batch256_per_sample_s=gpub, train_time_s=tr_s)
sol = res['batch600']['median']
print('加速比：GPU 单样本 %.0fx | CPU 单线程 %.0fx（对 600 组中位求解时间 %.1f s）'
      % (sol/gpu1, sol/cpu1, sol), flush=True)
json.dump(res, open(os.path.join(OUT, 'timing.json'), 'w'), indent=1)
print('saved', os.path.join(OUT, 'timing.json'))