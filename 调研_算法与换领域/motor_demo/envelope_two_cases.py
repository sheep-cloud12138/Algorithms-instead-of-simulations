# -*- coding: utf-8 -*-
"""Torque-speed envelopes for series Y / parallel Y / delta, two inductance levels."""
import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

I_MAX=1.0; V_Y=1.0; R=0.03; P=2; KT=1.0
LAM=KT/3.0

def make_cfg(ich):
    L=LAM/ich
    return {
     'series Y':   dict(kt=KT,   lam=LAM,   L=L,     R=R,      V=V_Y,           I=I_MAX),
     'parallel Y': dict(kt=KT/2, lam=LAM/2, L=L/4,   R=R/4,    V=V_Y,           I=I_MAX),
     'delta':      dict(kt=KT,   lam=LAM,   L=L,     R=R,      V=V_Y*np.sqrt(3), I=I_MAX/np.sqrt(3)),
    }

def tmax(c, w):
    kt,lam,Lc,Rc,V,I = c['kt'],c['lam'],c['L'],c['R'],c['V'],c['I']
    best=0.0
    for id_ in np.linspace(-I,I,601):
        a=(w*Lc)**2+Rc**2
        b=2*(-w*Lc*Rc*id_ + Rc*w*(Lc*id_+lam))
        cc=(Rc*id_)**2+(w*(Lc*id_+lam))**2-V**2
        disc=b*b-4*a*cc
        iq_v=(-b+np.sqrt(disc))/(2*a) if disc>0 else -1
        iq=min(np.sqrt(max(I*I-id_*id_,0)),max(iq_v,0))
        best=max(best,kt*iq)
    return best

def loss_at(c,w,TL):
    kt,lam,Lc,Rc,V,I=c['kt'],c['lam'],c['L'],c['R'],c['V'],c['I']
    iq=TL/kt
    if iq>I: return np.nan
    best=np.nan
    for id_ in np.linspace(-I,I,401):
        if id_*id_+iq*iq>I*I: continue
        vd=Rc*id_-w*Lc*iq; vq=Rc*iq+w*(Lc*id_+lam)
        if vd*vd+vq*vq<=V*V:
            L_=1.5*(id_*id_+iq*iq)*Rc
            if np.isnan(best) or L_<best: best=L_
    return best

def curves(ich, ws):
    cfgs=make_cfg(ich); out={}
    for n,c in cfgs.items():
        T=np.array([tmax(c,w) for w in ws])
        # base speed = first w where T < 0.999*T(0)
        T0=T[0]; idx=np.argmax(T<0.999*T0)
        wb=ws[idx] if idx>0 else ws[-1]
        out[n]=dict(T=T,wb=wb,cfg=c)
    return out

ws=np.linspace(0.02,12,700)
fig,axs=plt.subplots(2,2,figsize=(12,8))
summary={}
for row,(ich,label) in enumerate([(0.5,'high inductance: I_ch/I_max = 0.5'),
                                  (5.0,'low inductance: I_ch/I_max = 5')]):
    out=curves(ich,ws)
    wb0=out['series Y']['wb']
    for n,d in out.items():
        axs[row,0].plot(ws/wb0, d['T'], label=n, lw=2)
        axs[row,1].plot(ws/wb0, [loss_at(d['cfg'],w,0.3) for w in ws], label=n, lw=2)
        summary[(label,n)]=dict(base_speed=float(d['wb']), T0=float(d['T'][0]))
    axs[row,0].set_xlim(0,6); axs[row,0].set_ylim(0,1.05)
    axs[row,0].set_title('Torque envelope — '+label)
    axs[row,0].set_ylabel('torque / series rated'); axs[row,0].legend(); axs[row,0].grid(alpha=.3)
    axs[row,1].set_xlim(0,6)
    axs[row,1].set_title('Copper loss at 0.3 p.u. load — '+label)
    axs[row,1].set_ylabel('loss (p.u.)'); axs[row,1].legend(); axs[row,1].grid(alpha=.3)
for a in axs[1,:]: a.set_xlabel('speed / series base speed')
plt.tight_layout(); plt.savefig('envelope_two_cases.png',dpi=150)
print('saved envelope_two_cases.png')

# crossover: series vs parallel and series vs delta (high inductance case)
out=curves(0.5,ws); wb0=out['series Y']['wb']
for other in ['parallel Y','delta']:
    d=np.array(out['series Y']['T'])-np.array(out[other]['T'])
    k=np.argmax(d<0)
    print('high-Ich: series vs %s crossover at %.2f x base'%(other, ws[k]/wb0 if k else np.nan))
