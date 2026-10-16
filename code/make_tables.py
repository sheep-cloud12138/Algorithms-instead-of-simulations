"""把所有结果 JSON 汇总成 markdown 表格（避免手工转录出错）。"""
import json, os, numpy as np
P = os.path.expanduser('~/nc_run/p7')
CH = ['ux','uy','theta','p']
def L(t): print('\n' + t)
def row(*v): print('| ' + ' | '.join(str(x) for x in v) + ' |')
def hdr(*v): row(*v); print('|' + '---|'*len(v))

def j(name):
    p = os.path.join(P, name)
    return json.load(open(p)) if os.path.exists(p) else None

g = lambda x: '%.4f' % x if isinstance(x,(int,float)) else str(x)

d = j('p2_fixed.json')
if d:
    L('## 主表：U-Net（width 32，200 epoch，残差学习）')
    hdr('划分','通道','A 无 ε','B 有 ε','提升','什么都不做','B 恢复的修正能量','floor 的绝对量级')
    for sp, spn in [('extrap','外推（训 ε≤1.3、测 ε>1.3）'), ('interp','内插（按 ε 分层 2:1）')]:
        if sp not in d: continue
        for k, c in enumerate(CH):
            A = d[sp]['A_no_eps']['g'][k]; B = d[sp]['B_eps']['g'][k]; F = d[sp]['floor'][k]
            row(spn if k==0 else '', c, g(A), g(B), '%.2f×'%(A/B), g(F), '%.1f%%'%(100*(1-(B/F)**2)), g(F))

d = j('prior_ablation.json')
if d:
    L('## 先验形式消融（同一个 U-Net，只改输出端的 ε 结构）')
    hdr('划分','变体','ux','uy','θ','p','网络学到的 β（四通道）')
    for sp, spn in [('extrap','外推'), ('interp','内插')]:
        if sp not in d: continue
        for tag in ['B_plain_no_prior','P1_scaled_eps','P2_scaled_eps_quad']:
            if tag not in d[sp]: continue
            r = d[sp][tag]
            b = r.get('beta')
            bs = ' '.join('%.3f'%x for x in b) if b and any(abs(x)>1e-9 for x in b) else '—'
            row(spn, tag, *[g(x) for x in r['g']], bs)

d = j('fno_results.json')
if d:
    L('## 强基线：FNO2d（谱算子学习，modes 12，width 32，4 层，参数量 %.0fk）' % (d.get('n_params',0)/1e3))
    hdr('划分','变体','ux','uy','θ','p')
    for sp, spn in [('extrap','外推'), ('interp','内插')]:
        if sp not in d: continue
        for tag in ['A_no_eps','B_eps','C_prior']:
            if tag in d[sp]: row(spn, tag, *[g(x) for x in d[sp][tag]['g']])

d = j('learning_curve.json')
if d:
    L('## 学习曲线（内插测试集 200 组，3 个种子，均值±std）')
    hdr('训练样本量','变体','ux','uy','θ','p')
    print('| 0（什么都不做） | — | ' + ' | '.join('%.4f'%x for x in d['floor']) + ' |')
    for var, vn in [('B_eps_input','B：输入含 ε'), ('C_prior','C：只学 P，ε 不进网络')]:
        for n in sorted(d['curves'][var], key=int):
            m = d['curves'][var][n]['mean']; s = d['curves'][var][n]['std']
            row(n, vn, *['%.4f±%.4f'%(m[i],s[i]) for i in range(4)])

d = j('public_transfer.json')
if d:
    L('## 公开数据集（4000 组，ε≡1.2）外部验证与跨数据集迁移')
    if 'external_law' in d:
        e = d['external_law']
        hdr('通道','公开数据 ε=1.2 中位 R','p10','p90','本机 ε∈[1.1,1.3] 中位')
        for k,c in enumerate(CH):
            row(c, '%.4f'%e['R_pub_median'][k], '%.4f'%e['R_pub_p10'][k], '%.4f'%e['R_pub_p90'][k], '%.4f'%e['R_own_median'][k])
    for sp in ['extrap','interp']:
        if sp not in d: continue
        L('划分 %s（测试 n=%d；floor=%s）' % (sp, d[sp].get('n_test',0), ' '.join('%.4f'%x for x in d[sp]['floor'])))
        hdr('训练数据','ux','uy','θ','p')
        for tag in d[sp]:
            if tag in ('floor','n_test'): continue
            row(tag, *[g(x) for x in d[sp][tag]['g']])

d = j('aspect2.json')
if d:
    L('## 宽高比 2:1（另一几何，%d 组）' % d['n'])
    hdr('通道','2:1 几何 p','A','R²','单位方腔 p','单位方腔 A')
    ref = {'ux':(1.082,0.1468),'uy':(1.001,0.1934),'theta':(1.216,0.0097),'p':(1.121,0.1482)}
    for c in CH:
        row(c, '%.3f'%d['law'][c]['p'], '%.4f'%d['law'][c]['A'], '%.3f'%d['law'][c]['R2'], '%.3f'%ref[c][0], '%.4f'%ref[c][1])

d = j('timing.json')
if d:
    L('## 求解器耗时与推理耗时')
    hdr('项','中位','均值','p90','最大')
    for k,kn in [('batch600','求解：600 组'),('paired100','求解：配对集')]:
        if k in d: row(kn, '%.1f s'%d[k]['median'], '%.1f s'%d[k]['mean'], '%.1f s'%d[k]['p90'], '%.1f s'%d[k]['max'])
    i = d['inference']
    row('推理 GPU 单样本', '%.1f ms'%(i['gpu_batch1_s']*1e3), '', '', '')
    row('推理 GPU 批量', '%.2f ms/样本'%(i['gpu_batch256_per_sample_s']*1e3), '', '', '')
    row('推理 CPU 单线程', '%.1f ms'%(i['cpu_1thread_s']*1e3), '', '', '')
    sm = d['batch600']['median']
    print('\n加速比：单样本 %.0f×，批量 %.0f×' % (sm/i['gpu_batch1_s'], sm/i['gpu_batch256_per_sample_s']))