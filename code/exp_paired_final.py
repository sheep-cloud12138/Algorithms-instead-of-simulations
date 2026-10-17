"""配对集（同问题、5 个 ε）的最终分析 —— 用修正后的 p 通道。
(1) 零成本基线、(2) ε-blind 不可约下界、(3) 修正场形状的 ε 不变性、
(4) 配对集上独立标定的 ε-律，与 600 组随机壁面的结果对照。"""
import numpy as np, os, glob, re, json
DS = os.path.expanduser('~/nc_run/paired/dataset'); EPS = [0.4,0.8,1.2,1.6,1.9]
CH = ['ux','uy','theta','p']; LB = ['u_x','u_y','theta','p']
def toimg(v, c):
    ux = np.unique(np.round(c[:,0],12)); uy = np.unique(np.round(c[:,1],12))
    img = np.full((len(uy), len(ux)), np.nan)
    img[np.searchsorted(uy,np.round(c[:,1],12)), np.searchsorted(ux,np.round(c[:,0],12))] = v
    if len(ux) == 129: img = img[::2,::2]
    return img[:64,:64]
def deramp(f):
    n = f.shape[0]; A = np.column_stack([np.ones(n*n), np.repeat(np.linspace(0,1,n),n)])
    cc,*_ = np.linalg.lstsq(A, f.ravel(), rcond=None); return (f.ravel()-A@cc).reshape(n,n)
def fields(d, e):
    o=[]
    for c in ['ux','uy','T']:
        b = toimg(np.load(os.path.join(d,c+'_bouss.npy')).astype(np.float64), np.load(os.path.join(d,'coords_%s_bouss.npy'%c)))
        p = toimg(np.load(os.path.join(d,c+'_comp.npy')).astype(np.float64),  np.load(os.path.join(d,'coords_%s_comp.npy'%c)))
        o.append((b,p))
    pb = toimg(np.load(d+'/p_bouss_pert_raw.npy'), np.load(d+'/coords_p_bouss.npy'))
    pc = toimg(np.load(d+'/p_comp_pert_raw.npy'),  np.load(d+'/coords_p_comp.npy'))
    o.append((-deramp(pb), deramp(pc)))
    th = lambda T: (T-(1-e/2))/e
    return [o[0], o[1], [th(o[2][0]), th(o[2][1])], o[3]]
tags={}
for d in glob.glob(DS+'/c*_Ra*'):
    t=os.path.basename(d); m=re.match(r'(c\d+_Ra\S+?_\d{4})_e([\d.]+)$',t)
    if m and os.path.exists(os.path.join(d,'sample_000000','meta.json')):
        tags.setdefault(m.group(1),{})[float(m.group(2))]=os.path.join(d,'sample_000000')
cfgs=sorted(k for k,v in tags.items() if len(v)==len(EPS))
C={c:{} for c in cfgs}
for cfg in cfgs:
    for e in EPS: C[cfg][e]=fields(tags[cfg][e], e)
rl=lambda a,b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
cs=lambda a,b: float(a.ravel()@b.ravel()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
out={}
print('配对配置 %d 组 x %d 个 eps = %d 个样本'%(len(cfgs),len(EPS),len(cfgs)*len(EPS)))
print('\n=== 1. 零成本基线「什么都不做」：|Y-X|/|Y| 中位 ===')
print('%-6s'%'eps' + ''.join('%12s'%('R('+LB[i]+')') for i in range(4)))
tab_floor={}
for e in EPS:
    v=[rl(C[c][e][k][0], C[c][e][k][1]) for c in cfgs for k in range(4)]
    v=np.array(v).reshape(len(cfgs),4).T
    tab_floor[e]=[float(np.median(x)) for x in v]
    print('%-6.1f'%e + ''.join('%12.4f'%np.median(x) for x in v))
out['floor']=tab_floor
print('\n=== 2. ε-blind 模型的不可约下界（配对集，中位） ===')
print('%-6s %-32s %-32s'%('eps','A: 只在 eps=1.2 训练','B: 最优 ε-blind = 各 ε 均值'))
tab_b={}
for e in EPS:
    ta=[]; tb=[]
    for c in cfgs:
        y=[C[c][e][k][1] for k in range(4)]; y0=[C[c][1.2][k][1] for k in range(4)]
        ym=[np.mean([C[c][ee][k][1] for ee in EPS],axis=0) for k in range(4)]
        ta.append([rl(y0[k],y[k]) for k in range(4)]); tb.append([rl(ym[k],y[k]) for k in range(4)])
    ta=np.median(np.array(ta),0); tb=np.median(np.array(tb),0)
    tab_b[e]={'A':ta.tolist(),'B':tb.tolist()}
    print('%-6.1f A: %-28s B: %s'%(e,' '.join('%.4f'%x for x in ta),' '.join('%.4f'%x for x in tb)))
out['eps_blind']=tab_b
print('\n=== 2b. eps-blind 迁移矩阵：拿 eps=t 的解当 eps 的预测（逐对精确值，对角线=0）===')
tab_tr = {}
for e_te in EPS:
    for e_tr in EPS:
        v = [rl(C[c][e_tr][k][1], C[c][e_te][k][1]) for c in cfgs for k in range(4)]
        v = np.array(v).reshape(len(cfgs), 4)
        tab_tr['%.1f_from_%.1f' % (e_te, e_tr)] = [float(np.median(v[:, k])) for k in range(4)]
    print('  测试 eps=%.1f  ' % e_te + ' | '.join('用 %.1f 的解: %s' % (e_tr,
          ' '.join('%.3f' % tab_tr['%.1f_from_%.1f' % (e_te, e_tr)][k] for k in range(4))) for e_tr in EPS))
out['transfer_matrix'] = tab_tr

print('\n=== 3. 修正场形状随 eps 的不变性（与 eps=1.2 的 cos，中位）===')
tab_cs={}
for k,name in enumerate(CH):
    v=[cs(C[c][e][k][1]-C[c][e][k][0], C[c][1.2][k][1]-C[c][1.2][k][0]) for c in cfgs for e in EPS]
    v=np.array(v).reshape(len(cfgs),len(EPS))
    tab_cs[name]=[float(np.median(v[:,j])) for j in range(len(EPS))]
    print('  %-6s'%name + ''.join('%9.3f'%np.median(v[:,j]) for j in range(len(EPS))))
out['cos_shape']=tab_cs
print('\n=== 4. 配对集上独立标定：逐配置 R(ε) 的幂律拟合（中位） ===')
tab_p={}
for k,name in enumerate(CH):
    ps=[]; As=[]; R2s=[]; Rs=[]
    for c in cfgs:
        r=np.array([rl(C[c][e][k][1],C[c][e][k][0]) for e in EPS]); Rs.append(r)
        p,i0,rv,pv,se=np.polyfit(np.log(EPS),np.log(r),1)[0],0,0,0,0
        from scipy import stats
        sl,ic,rv,pv,se = stats.linregress(np.log(EPS), np.log(r))
        ps.append(sl); As.append(np.exp(ic)); R2s.append(rv**2)
    Rs=np.array(Rs)
    print('  %-6s p=%.3f   A=%.4f   R²中位 %.3f   R 中位 %s'%(name,np.median(ps),np.median(As),np.median(R2s),' '.join('%.4f'%x for x in np.median(Rs,0))))
    tab_p[name]=dict(p=float(np.median(ps)), A=float(np.median(As)), R2=float(np.median(R2s)), R=[float(x) for x in np.median(Rs,0)])
out['paired_power']=tab_p
json.dump(out, open(os.path.expanduser('~/nc_run/p7/paired_final.json'),'w'), indent=1)
print('\nsaved ~/nc_run/p7/paired_final.json')