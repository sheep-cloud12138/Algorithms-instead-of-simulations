# -*- coding: utf-8 -*-
"""Torque-speed envelope with voltage/current limits and field weakening."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

I_MAX=1.0; V_Y=1.0; R=0.03; I_CH=0.5; P=2
KT=1.0                      # series torque per phase current
LAM=KT/3.0                  # lambda_pm = kt/(3p)*... per-unit
L= LAM/I_CH

CFG = {
 'series Y':   dict(kt=KT,   lam=LAM,   L=L,     R=R,      V=V_Y,      I=I_MAX),
 'parallel Y': dict(kt=KT/2, lam=LAM/2, L=L/4,   R=R/4,    V=V_Y,      I=I_MAX),
 'delta':      dict(kt=KT, lam=LAM, L=L, R=R,   V=V_Y*np.sqrt(3), I=I_MAX/np.sqrt(3)),
}

def tmax(cfg, w):
    kt,lam,Lc,Rc,V,I = cfg['kt'],cfg['lam'],cfg['L'],cfg['R'],cfg['V'],cfg['I']
    best=0.0; best_id=0.0
    for id_ in np.linspace(-I,I,801):
        # voltage limit: (R id - w L iq)^2 + (R iq + w(L id + lam))^2 <= V^2
        a=(w*Lc)**2+Rc**2
        b=2*(-w*Lc*Rc*id_ + Rc*w*(Lc*id_+lam))
        c=(Rc*id_)**2+(w*(Lc*id_+lam))**2-V**2
        disc=b*b-4*a*c
        iq_v = (-b+np.sqrt(disc))/(2*a) if disc>0 else -1
        iq=min(np.sqrt(max(I*I-id_*id_,0)), max(iq_v,0))
        T=kt*iq
        if T>best: best=T; best_id=id_
    return best,best_id

ws=np.linspace(0.02,8,400)
fig,ax=plt.subplots(1,2,figsize=(12,4.8))
for name,c in CFG.items():
    T=np.array([tmax(c,w)[0] for w in ws])
    ax[0].plot(ws,T,label=name,lw=2)
ax[0].set_xlim(0,6); ax[0].set_ylim(0,1.05)
ax[0].set_xlabel('speed / series base speed'); ax[0].set_ylabel('torque / series rated torque')
ax[0].set_title('Torque-speed envelope (V/I limits + field weakening)')
ax[0].grid(alpha=.3); ax[0].legend()

# copper loss at a load torque of 0.3 p.u., choose min-loss config
TL=0.3
for name,c in CFG.items():
    Pcu=[]
    for w in ws:
        # find min current that gives TL, respecting voltage
        kt,lam,Lc,Rc,V,I=c['kt'],c['lam'],c['L'],c['R'],c['V'],c['I']
        best=np.nan
        for id_ in np.linspace(-I,I,601):
            iq=TL/kt
            if id_*id_+iq*iq>I*I: continue
            vd=Rc*id_-w*Lc*iq; vq=Rc*iq+w*(Lc*id_+lam)
            if vd*vd+vq*vq<=V*V:
                loss=1.5*(id_*id_+iq*iq)*Rc
                if np.isnan(best) or loss<best: best=loss
        Pcu.append(best)
    ax[1].plot(ws,Pcu,label=name,lw=2)
ax[1].set_xlim(0,6); ax[1].set_xlabel('speed / series base speed')
ax[1].set_ylabel('copper loss (p.u.)'); ax[1].set_title('Copper loss at load = 0.3 p.u.')
ax[1].grid(alpha=.3); ax[1].legend()
plt.tight_layout(); plt.savefig('envelope.png',dpi=150)
print('saved envelope.png')

for w in [1,1.5,2,3,4]:
    print('w=%.1f  '%w + '  '.join('%s %.3f'%(n,tmax(c,w)[0]) for n,c in CFG.items()))
