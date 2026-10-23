# -*- coding: utf-8 -*-
"""Extract lambda_pm, L, R and I_ch for the demo motor."""
import sys, numpy as np
sys.path.insert(0, r'D:\Desktop\算法代替仿真\调研_算法与换领域\motor_demo')
from motor_sweep import solve_case, MU0

NR, NT = 140, 320
L_STACK = 0.10
N_PHASE = 160          # 4 slots/phase x 40 turns
P = 2
R_G = 0.0305

f = solve_case(0.0, 1.0, NR=NR, NT=NT, I_phase=0.0, pm_on=True, return_fields=True)
r, th, Br = f['r'], f['th'], f['Br']
k = int(np.argmin(np.abs(r-R_G)))
br = Br[k]
# fundamental over one electrical period pi (p=2)
a1 = (1/np.pi)*np.trapezoid(br*np.cos(2*th), th)
b1 = (1/np.pi)*np.trapezoid(br*np.sin(2*th), th)
B1 = np.hypot(a1,b1)
print('gap r=%.2f mm  Br rms=%.3f  fundamental B1=%.3f T (a=%.3f b=%.3f)'%(r[k]*1e3,np.sqrt((br**2).mean()),B1,a1,b1))

lam = 2*N_PHASE*B1*r[k]*L_STACK/P
print('lambda_pm = %.4f Wb'%lam)

# analytical dq inductance (surface PM, airgap + magnet)
g = 0.0310-0.0295
lm = 0.004
g_eff = g + lm/1.05
L = 1.5*(4/np.pi)*MU0*N_PHASE**2*r[k]*L_STACK/(P**2*g_eff)
print('L_dq(analytical) = %.2f mH   (g_eff=%.3f mm)'%(L*1e3,g_eff*1e3))

# resistance: slot area, fill 0.45, mean turn 2*(stack+2*end)
SLOT_AREA=(0.0475-0.0355)*(2*np.pi/12*0.52)*(0.0355+0.0475)/2
A_cond=0.45*SLOT_AREA/40
l_turn=2*(L_STACK+0.06)
R=1.72e-8*N_PHASE*l_turn/A_cond
print('R_phase = %.3f ohm  (A_cond=%.3f mm^2, l_turn=%.3f m)'%(R,A_cond*1e6,l_turn))

Ich=lam/L
print('I_ch = %.1f A'%Ich)
for Imax in [30,60,120,200]:
    print('  I_max=%3d A -> I_ch/I_max = %.2f'%(Imax,Ich/Imax))
