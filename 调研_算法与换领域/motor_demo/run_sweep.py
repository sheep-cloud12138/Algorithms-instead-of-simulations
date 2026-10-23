# -*- coding: utf-8 -*-
import csv, numpy as np
from concurrent.futures import ProcessPoolExecutor
from motor_sweep import solve_case

CONFIGS = [('series',1.0),('parallel',0.5),('delta',1/np.sqrt(3))]
THETAS = np.arange(12)*7.5
I_REF = 60.0

def job(a):
    cfg,s,th = a
    r = solve_case(th, s, NR=60, NT=120, I_phase=I_REF)
    return dict(config=cfg, scale=s, theta=th, T=r['T_lorentz'], Bgap=r['Bgap'],
                Biron=r['Bmax_iron'], iters=r['iters'], rel=r['rel'])

if __name__ == '__main__':
    jobs=[(c,s,t) for c,s in CONFIGS for t in THETAS]
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows=list(ex.map(job,jobs))
    with open('sweep_3x12.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('config     meanT   minT   maxT  ripple  ratio  Bgap')
    base=np.mean([r['T'] for r in rows if r['config']=='series'])
    for c,_ in CONFIGS:
        T=np.array([r['T'] for r in rows if r['config']==c])
        print('%-10s %6.2f %6.2f %6.2f  %5.1f%%  %.3f  %.3f'%(
            c,T.mean(),T.min(),T.max(),100*(T.max()-T.min())/T.mean(),T.mean()/base,
            np.mean([r['Bgap'] for r in rows if r['config']==c])))
    print('saved sweep_3x12.csv')
