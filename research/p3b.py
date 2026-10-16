import csv,numpy as np,io,sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
R=list(csv.DictReader(open("data/train_legacy/per_sample.csv",encoding="utf-8")))
g=lambda c: np.array([float(r[c]) for r in R])
ra=g("Ra"); lr=np.log10(ra)
print("log10(Ra) 与各量（log 后）的相关：")
for c in ["sqrt_RaPr","vel_factor","umax_in","umax_tgt","p_rms_in","p_rms_tgt","rms_r_B","a_B","b_B","s_B","m_in"]:
    v=g(c); m=np.isfinite(v)&(v>0)
    print("  %-12s corr=%+.5f   min=%.4g max=%.4g"%(c,np.corrcoef(lr[m],np.log10(v[m]))[0,1],v[m].min(),v[m].max()))
print()
print("检验解析关系：把已知因子除掉以后，剩下的还随 Ra 变吗？")
for name,num,den in [("p_rms_in",g("p_rms_in"),ra*np.sqrt(ra*0.71)),("umax_in",g("umax_in"),np.sqrt(ra*0.71))]:
    q=num/den; m=np.isfinite(q)&(q>0)
    print("  %-10s 去掉因子后比值 %.4g~%.4g（波动 %.1f%%），corr(log比值, logRa)=%+.3f"
          %(name,q[m].min(),q[m].max(),100*(q[m].max()-q[m].min())/q[m].mean(),np.corrcoef(np.log10(q[m]),lr[m])[0,1]))
print()
print("  参考：p_rms_in 的 log 与 logRa 相关 = %+.5f"%np.corrcoef(lr,np.log10(g("p_rms_in")))[0,1])
print("  参考：umax_in 的 log 与 logRa 相关 = %+.5f"%np.corrcoef(lr,np.log10(g("umax_in")))[0,1])