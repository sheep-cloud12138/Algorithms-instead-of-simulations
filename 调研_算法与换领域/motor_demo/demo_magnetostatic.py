# -*- coding: utf-8 -*-
"""2D 非线性静磁学 (A_z, 极坐标有限体积) — 4极表贴式永磁电机简化截面
方程: -div(nu grad A) = J + (curl M)_z
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, time, sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MU0 = 4*np.pi*1e-7
RIN, ROUT = 0.005, 0.065
NR, NT = int(os.environ.get("NR", 60)), int(os.environ.get("NT", 120))
r_edges = np.linspace(RIN, ROUT, NR+1)
dr = (ROUT-RIN)/NR; dth = 2*np.pi/NT
r_c = 0.5*(r_edges[:-1]+r_edges[1:]); th_c = (np.arange(NT)+0.5)*dth

R_SHAFT, R_ROTOR, R_BORE = 0.012, 0.0295, 0.0310
R_SLOT_IN, R_SLOT_OUT = 0.0355, 0.0475
N_SLOT = 12; SLOT_W = 2*np.pi/N_SLOT*0.52
N_POLE = 4; PPAIR = N_POLE//2

r2 = r_c[:,None]; th2 = th_c[None,:]

is_air = np.zeros((NR,NT), bool)
is_air |= (r2 >= R_ROTOR) & (r2 <= R_BORE)
slot_mask = (r2 >= R_SLOT_IN) & (r2 <= R_SLOT_OUT) & \
            (np.abs(np.mod(th2 + np.pi/(2*N_SLOT), 2*np.pi/N_SLOT) - np.pi/N_SLOT) < SLOT_W/2)
is_air |= slot_mask
is_air |= (r2 < R_SHAFT)
in_mag = (r2 >= R_ROTOR-0.004) & (r2 < R_ROTOR)
is_mag = in_mag & (np.abs(np.mod(th2, 2*np.pi/N_POLE) - np.pi/N_POLE) < np.pi/N_POLE*0.80)
is_air &= ~is_mag
is_iron = ~is_air & ~is_mag

# ---- 绕组: 12槽, 4极正弦电流片近似 ----
I_AMP, N_TURN = 200.0, 40
SLOT_AREA = (R_SLOT_OUT-R_SLOT_IN)*SLOT_W*(R_SLOT_IN+R_SLOT_OUT)/2
J_slot = I_AMP*N_TURN/SLOT_AREA
th_slot = (np.arange(NT)+0.5)*dth
j_slot = np.cos(PPAIR*th_slot)
J = np.zeros((NR,NT))
for j in range(NT):
    J[slot_mask[:,j], j] = J_slot*j_slot[j]

# ---- 永磁体: 径向磁化, 交替 ----
BR_PM = 1.2
pole_sign = np.tanh(np.sin(PPAIR*th2)/0.12)
M_r = np.where(is_mag, BR_PM/MU0*pole_sign, 0.0)
src_pm = -np.gradient(M_r, dth, axis=1)/r_c[:,None]
SRC = J + src_pm

MU_R0, B_SAT, H0 = 4000.0, 1.6, 6000.0
def mu_r_of_B(B):
    B = np.maximum(B, 1e-9)
    H = B/(MU0*MU_R0) + (B/B_SAT)**6 * H0
    return B/(MU0*H)

idx = np.arange(NR*NT).reshape(NR,NT)
def assemble(nu_cell):
    nu_r = 0.5*(nu_cell[:-1,:]+nu_cell[1:,:])
    nu_t = 0.5*(nu_cell+np.roll(nu_cell,-1,axis=1))
    # 通量/体积: radial  nu*r_face/dr^2/r_c ; azimuthal nu/(r_c^2 dth^2)
    rf = r_edges[1:-1,None]
    cr_p = nu_r*rf/(r_c[:-1,None]*dr*dr)
    cr_m = nu_r*rf/(r_c[1:,None]*dr*dr)
    ct = nu_t/(r_c[:,None]**2*dth*dth)
    Ar_p = np.zeros((NR,NT)); Ar_p[:-1,:] = cr_p
    Ar_m = np.zeros((NR,NT)); Ar_m[1:,:]  = cr_m
    At_p = ct; At_m = np.roll(ct,1,axis=1)
    diag = Ar_p+Ar_m+At_p+At_m
    rows=[idx.ravel()]; cols=[idx.ravel()]; vals=[diag.ravel()]
    for off,d in ((np.roll(idx,-1,axis=0),Ar_p),(np.roll(idx,1,axis=0),Ar_m),
                  (np.roll(idx,-1,axis=1),At_p),(np.roll(idx,1,axis=1),At_m)):
        m=d!=0; rows.append(idx[m]); cols.append(off[m]); vals.append((-d)[m])
    A=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
                    shape=(NR*NT,NR*NT)).tocsc()
    b=SRC.ravel().copy()
    A=A.tolil()
    for rr in (idx[-1,:], idx[0,:]):
        A[rr,:]=0
        for k in rr: A[k,k]=1.0
        b[rr]=0.0
    return A.tocsc(), b

nu = np.ones((NR,NT))/MU0
t0=time.time()
for it in range(60):
    A,b = assemble(nu)
    Az = spl.spsolve(A,b).reshape(NR,NT)
    Br = np.gradient(Az, dth, axis=1)/r_c[:,None]
    Bth = -np.gradient(Az, dr, axis=0)
    Bmag = np.sqrt(Br**2+Bth**2)
    nu_new = 1.0/(MU0*mu_r_of_B(Bmag))
    nu_new[is_air|is_mag] = 1.0/MU0
    rel = np.linalg.norm(nu_new-nu)/max(np.linalg.norm(nu_new),1e-30)
    nu = 0.7*nu + 0.3*nu_new
    if it%5==0 or rel<1e-7:
        print("iter %2d  rel=%.2e  |B|max(iron)=%.3f T  t=%.1fs" % (it, rel, Bmag[is_iron].max(), time.time()-t0), flush=True)
    if rel<1e-7: break

print("\n完成: %.1f s,  网格 %dx%d = %d 单元" % (time.time()-t0, NR, NT, NR*NT))
gap = np.broadcast_to(((r2>=R_ROTOR)&(r2<=R_BORE)), (NR,NT))
print("气隙平均 |B| = %.4f T" % Bmag[gap].mean())
print("铁芯最大 |B| = %.3f T" % Bmag[is_iron].max())
print("永磁体最大 |B| = %.3f T" % Bmag[is_mag].max())
tag = os.environ.get("TAG","coarse")
np.savez(f"motor_field_{tag}.npz", r=r_c, th=th_c, Az=Az, Bmag=Bmag, Br=Br, Bth=Bth,
         is_iron=is_iron, is_mag=is_mag, is_air=is_air, J=J, slot_mask=slot_mask)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
Rg,Tg = np.meshgrid(r_edges, np.arange(NT+1)*dth, indexing="ij")
Xg,Yg = Rg*np.cos(Tg), Rg*np.sin(Tg)
mat = np.where(is_mag, 2.0, np.where(is_iron, 1.0, 0.0))
fig, ax = plt.subplots(1,2,figsize=(12,5.2))
c=ax[0].pcolormesh(Xg*1e3,Yg*1e3,Bmag,shading="auto",cmap="inferno")
ax[0].set_aspect("equal"); ax[0].set_title("|B| (T)"); ax[0].set_xlabel("x (mm)")
plt.colorbar(c,ax=ax[0],shrink=0.85)
c2=ax[1].pcolormesh(Xg*1e3,Yg*1e3,mat,shading="auto",cmap="viridis")
ax[1].set_aspect("equal"); ax[1].set_title("material (1=iron, 2=PM, 0=air)"); ax[1].set_xlabel("x (mm)")
plt.tight_layout(); plt.savefig(f"motor_field_{tag}.png",dpi=150)
print("saved motor_field_%s.png" % tag)
