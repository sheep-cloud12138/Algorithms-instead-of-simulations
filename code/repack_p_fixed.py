"""修复压力通道：两个成员的原始解里，可压缩侧带静水斜坡、且弱形式符号相反。
按公开数据集的同一配方（去均值 -> 去掉 y 的线性最小二乘拟合 -> 输入侧反号）重建 p 通道，
其余三个通道逐位保留。输出 pairs600_fixedp.npz / pairs_paired_fixedp.npz。"""
import numpy as np, json, os, glob

def deramp(f):
    n = f.shape[0]
    A = np.column_stack([np.ones(n*n), np.repeat(np.linspace(0,1,n), n)])
    c, *_ = np.linalg.lstsq(A, f.ravel(), rcond=None)
    return (f.ravel() - A@c).reshape(n, n)

def scatter(v, c, N):
    ux = np.unique(np.round(c[:,0],12)); uy = np.unique(np.round(c[:,1],12))
    col = np.searchsorted(ux, np.round(c[:,0],12)); row = np.searchsorted(uy, np.round(c[:,1],12))
    img = np.full((N,N), np.nan); img[row,col] = v
    if N == 129: img = img[::2, ::2]
    return img[:64, :64]

def p_pair(d):
    pb = scatter(np.load(d+'/p_bouss_pert_raw.npy'), np.load(d+'/coords_p_bouss.npy'), 65)
    pc = scatter(np.load(d+'/p_comp_pert_raw.npy'),  np.load(d+'/coords_p_comp.npy'),  65)
    return -deramp(pb).astype(np.float32), deramp(pc).astype(np.float32)

def do(pack_in, dirs, pack_out):
    z = np.load(pack_in, allow_pickle=True)
    X = z['X'].copy(); Y = z['Y'].copy(); meta = json.loads(str(z['meta']))
    assert len(dirs) == X.shape[0], (len(dirs), X.shape)
    dif = 0.0
    for i, d in enumerate(dirs):
        px, py = p_pair(d)
        dif = max(dif, float(np.abs(X[i,3]-px).max()))
        X[i,3] = px; Y[i,3] = py
    print('%s: %d 组重建完成（旧 p 与新 p 最大差 %.3f，说明确实变了）' % (os.path.basename(pack_out), len(dirs), dif))
    np.savez_compressed(pack_out, X=X, Y=Y, meta=np.array(json.dumps(meta)))
    return X, Y, meta

B = os.path.expanduser('~/nc_run')
d600 = sorted(glob.glob(B+'/batch/dataset/sample_*'))
d600 = sorted(d600, key=lambda s: int(os.path.basename(s).split('_')[1]))
X, Y, meta = do(B+'/p2/pairs600.npz', d600, B+'/p2/pairs600_fixedp.npz')

dp = sorted(glob.glob(B+'/paired/dataset/*/sample_000000'))
do(B+'/p4/pairs_paired.npz', dp, B+'/p4/pairs_paired_fixedp.npz')

# 修正后的 p 通道律与硬下界
eps = np.array([m['eps'] for m in meta]); Ra = np.array([m['Ra'] for m in meta])
D = Y - X
for k, name in [(3, 'p(修正后)')]:
    R = np.linalg.norm(D[:,k].reshape(len(eps),-1), axis=1)/np.linalg.norm(Y[:,k].reshape(len(eps),-1), axis=1)
    A1 = np.column_stack([eps, np.ones_like(eps)]); c1, *_ = np.linalg.lstsq(A1, R, rcond=None)
    r = R - A1@c1; r2 = 1-(r**2).sum()/((R-R.mean())**2).sum()
    A2 = np.column_stack([eps, eps**2, np.ones_like(eps)]); c2, *_ = np.linalg.lstsq(A2, R, rcond=None)
    print('%-10s 线性 R=%.4f*eps%+.4f (R2=%.3f) | 二次 a=%.4f c=%+.4f b=%+.4f' %
          (name, c1[0], c1[1], r2, c2[0], c2[1], c2[2]))
    s = eps <= 0.5
    c3, *_ = np.linalg.lstsq(np.column_stack([eps[s], np.ones(s.sum())]), R[s], rcond=None)
    print('           仅 eps<=0.5 (n=%d): a=%.4f b=%+.5f' % (s.sum(), c3[0], c3[1]))
    # 平方根律对照
    for form, F in [('sqrt(eps)', np.column_stack([np.sqrt(eps), np.ones_like(eps)])),
                    ('eps^0.75',   np.column_stack([eps**0.75, np.ones_like(eps)]))]:
        cf, *_ = np.linalg.lstsq(F, R, rcond=None); rr = R-F@cf
        print('           形式 %-9s R2=%.3f  截距%+.4f' % (form, 1-(rr**2).sum()/((R-R.mean())**2).sum(), cf[1]))