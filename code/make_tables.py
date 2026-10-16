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

# ================= 统一口径重跑（harness.py + exp_*_final.py）=================
def fmt_ms(r):
    m, sd = r['mean'], r['std']
    return ['%.4f±%.4f' % (m[i], sd[i]) for i in range(4)]

d = j('exp_ml_final.json')
if d:
    L('## 统一消融（U-Net width 32，200 epoch，3 个种子，归一化统计量只用训练子集）')
    print('协议：%s' % d.get('note', ''))
    hdr('划分', '变体', 'u_x', 'u_y', 'θ', 'p', '学到的 β')
    for sp, spn in [('extrap', '外推'), ('interp', '内插')]:
        if sp not in d: continue
        for tag in ['A_eps_blind', 'B_eps_input', 'P1_eps_in_x_eps', 'P2_eps_in_x_quad', 'P3_eps_not_in_net']:
            if tag not in d[sp]: continue
            r = d[sp][tag]
            b = r.get('beta')
            bs = '—'
            if b and isinstance(b[0], list) and any(abs(x) > 1e-9 for x in b[0]): bs = ' '.join('%.3f' % x for x in b[0])
            row(spn if tag.startswith('A') else '', tag, *fmt_ms(r), bs)
        row('', 'floor（什么都不做）', *['%.4f' % x for x in d[sp]['floor']], '')

d = j('exp_fno_final.json')
if d:
    L('## 强基线 FNO2d（%0.0fk 参数，同协议、同划分、3 个种子）' % (d.get('n_params', 0) / 1e3))
    hdr('划分', '变体', 'u_x', 'u_y', 'θ', 'p')
    for sp, spn in [('extrap', '外推'), ('interp', '内插')]:
        if sp not in d: continue
        for tag in ['A_eps_blind', 'B_eps_input', 'P3_eps_not_in_net']:
            if tag not in d[sp]: continue
            row(spn if tag.startswith('A') else '', tag, *fmt_ms(d[sp][tag]))
        row('', 'floor', *['%.4f' % x for x in d[sp]['floor']])

d = j('exp_lc_final.json')
if d:
    L('## 学习曲线（3 个种子，归一化统计量只用抽中的 n 个样本）')
    hdr('n', '变体', 'u_x', 'u_y', 'θ', 'p')
    print('| 0 | 什么都不做 | ' + ' | '.join('%.4f' % x for x in d['floor']) + ' |')
    for var in ['B_eps_input', 'P3_eps_not_in_net']:
        if var not in d['curves']: continue
        for n in sorted(d['curves'][var], key=int):
            row(n, var, *fmt_ms(d['curves'][var][n]))

d = j('pod_rom.json')
if d:
    L('## 经典 ROM 基线：POD 基 + 线性回归（k 模态覆盖训练集 99.9% 能量）')
    print('m1 = 只用 (logRa)；m2 = (logRa, ε, ε·logRa)；m3 = m2 + X 的 POD 系数（信息量与网络相同）')
    hdr('划分', '通道', 'k', 'm1 ε-blind', 'm2 ε-aware', 'm3 +X', 'floor')
    for sp, spn in [('extrap', '外推'), ('interp', '内插')]:
        if sp not in d: continue
        for i, c in enumerate(CH):
            r = d[sp]['ch'][c]
            row(spn if i == 0 else '', c, r['k'], '%.4f' % r['m1'], '%.4f' % r['m2'], '%.4f' % r['m3'],
                '%.4f' % d[sp]['floor'][i])

d = j('shallow_baselines.json')
if d:
    L('## 浅层基线：线性岭回归 / RBF 核岭回归（输入与网络相同）')
    hdr('划分', '模型', 'u_x', 'u_y', 'θ', 'p')
    for sp, spn in [('extrap', '外推'), ('interp', '内插')]:
        if sp not in d: continue
        row(spn, 'floor', *['%.4f' % x for x in d[sp]['floor']])
        for tag in ['ridge_blind', 'krr_blind', 'ridge_eps', 'krr_eps']:
            if tag in d[sp]['models']: row('', tag, *['%.4f' % x for x in d[sp]['models'][tag]])

d = j('public_legacy_transfer.json')
if d:
    L('## 跨 BC 族迁移：公开 5000 组经典方腔（ε≡1.2）训练 -> 本机随机壁面测试')
    hdr('划分', '训练数据', 'u_x', 'u_y', 'θ', 'p')
    for sp, spn in [('extrap', '外推'), ('interp', '内插')]:
        if sp not in d: continue
        row(spn, 'floor', *['%.4f' % x for x in d[sp]['floor']])
        for tag in ['legacy_naive', 'legacy_prior', 'own_prior']:
            if tag in d[sp]: row('', tag, *['%.4f' % x for x in d[sp][tag]['g']])

d = j('audit_v2.json')
if d:
    L('## 稳健性审计（audit_v2.json）')
    print('幂律指数 p 随拟合窗口（%s）：' % '627 组')
    hdr('窗口', *CH)
    for k in sorted(d['power_windows']):
        row(k, *['%.3f±%.3f' % (d['power_windows'][k][c]['p'], d['power_windows'][k][c]['se']) for c in CH])
    print('\n二次项 κ = c/a：' + '  '.join('%s %.2f' % (c, d['quadratic_kappa'][c]['kappa']) for c in CH))
    print('压力通道配对（逐样本中位）：')
    hdr('数据集', '|X_p|/|Y_p|', 'cos')
    for k in d['pressure']:
        row(k, '%.3f' % d['pressure'][k]['ratio_median'], '%+.3f' % d['pressure'][k]['cos_median'])
    print('\nû 中位：' + '  '.join('%s %.4f' % (k, v) for k, v in d['uhat_median'].items()))
    print('经典方腔/公开随机壁面 +%.0f%%，/本机 +%.0f%%' % (d['uhat_pct_vs_public_train'], d['uhat_pct_vs_ours']))
    print('legacy/train 的 R 中位之比：' + '  '.join('%s %.2f' % (c, d['ratio_legacy_over_train'][c]) for c in CH))
    print('2:1 几何指数与单位方腔之差：' + '  '.join('%s %.1fσ' % (c, d['aspect2_sigma'][c]) for c in CH))
