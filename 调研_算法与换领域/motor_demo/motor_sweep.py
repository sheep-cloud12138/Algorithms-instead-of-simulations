# -*- coding: utf-8 -*-
"""Parameterized 2D magnetostatic solver for winding-configuration sweep."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, time, sys, json

MU0 = 4*np.pi*1e-7
RIN, ROUT = 0.005, 0.065
R_SHAFT, R_ROTOR, R_BORE = 0.012, 0.0295, 0.0310
R_SLOT_IN, R_SLOT_OUT = 0.0355, 0.0475
N_SLOT, N_POLE = 12, 4
PPAIR = N_POLE//2
SLOT_W = 2*np.pi/N_SLOT*0.52
MU_R0, B_SAT, H0 = 4000.0, 1.6, 6000.0

def mu_r_of_B(B):
    B = np.maximum(B, 1e-9)
    H = B/(MU0*MU_R0) + (B/B_SAT)**6 * H0
    return B/(MU0*H)

def build(NR, NT):
    r_edges = np.linspace(RIN, ROUT, NR+1)
    dr = (ROUT-RIN)/NR; dth = 2*np.pi/NT
    r_c = 0.5*(r_edges[:-1]+r_edges[1:]); th_c = (np.arange(NT)+0.5)*dth
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
    return dict(r_edges=r_edges, dr=dr, dth=dth, r_c=r_c, th_c=th_c,
                r2=r2, th2=th2, is_air=is_air, slot_mask=slot_mask,
                is_mag=is_mag, is_iron=is_iron)

def solve_case(theta_r_deg, mmf_scale, NR=60, NT=120, I_phase=60.0, N_TURN=40, pm_on=True, return_fields=False):
    g = build(NR, NT)
    r_c, th_c = g['r_c'], g['th_c']
    r2, th2 = g['r2'], g['th2']
    dr, dth = g['dr'], g['dth']
    th_r = np.deg2rad(theta_r_deg)

    # PM: radial, rotated by theta_r
    pole_sign = np.tanh(np.sin(PPAIR*(th2-th_r))/0.12)
    M_r = np.where(g['is_mag'], (1.2/MU0 if pm_on else 0.0)*pole_sign, 0.0)
    src_pm = -np.gradient(M_r, dth, axis=1)/r_c[:,None]

    # current sheet at q-axis (max at theta = theta_r)
    SLOT_AREA = (R_SLOT_OUT-R_SLOT_IN)*SLOT_W*(R_SLOT_IN+R_SLOT_OUT)/2
    J0 = I_phase*N_TURN/SLOT_AREA*mmf_scale
    j_slot = np.sin(PPAIR*(th_c-th_r))
    J = np.zeros((NR,NT))
    for j in range(NT):
        J[g['slot_mask'][:,j], j] = J0*j_slot[j]
    SRC = J + src_pm

    idx = np.arange(NR*NT).reshape(NR,NT)
    def assemble(nu_cell):
        nu_r = 0.5*(nu_cell[:-1,:]+nu_cell[1:,:])
        nu_t = 0.5*(nu_cell+np.roll(nu_cell,-1,axis=1))
        rf = g['r_edges'][1:-1,None]
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
    for it in range(60):
        A,b = assemble(nu)
        Az = spl.spsolve(A,b).reshape(NR,NT)
        Br = np.gradient(Az, dth, axis=1)/r_c[:,None]
        Bth = -np.gradient(Az, dr, axis=0)
        Bmag = np.hypot(Br,Bth)
        nu_new = 1.0/(MU0*mu_r_of_B(Bmag))
        nu_new[g['is_air']|g['is_mag']] = 1.0/MU0
        rel = np.linalg.norm(nu_new-nu)/max(np.linalg.norm(nu_new),1e-30)
        nu = 0.7*nu + 0.3*nu_new
        if rel < 2e-4 and it > 20:
            break

    # Lorentz torque on conductors (robust to mesh)
    T = float(np.sum((r_c[:,None]*J*Br)*(r_c[:,None]*dr*dth)*g['slot_mask']))
    # Maxwell stress averaged over gap radii
    sel = (r_c>0.0297)&(r_c<0.0311)
    Tm = float(np.mean([ri*ri/MU0*np.sum(Br[k]*Bth[k])*dth for k,ri in enumerate(r_c) if sel[k]]))
    gap = (r_c>0.0295)&(r_c<0.0310)
    out = dict(theta_r=theta_r_deg, mmf=mmf_scale, T_lorentz=T, T_maxwell=Tm,
               Bgap=float(Bmag[gap].mean()), Bmax_iron=float(Bmag[g['is_iron']].max()),
               iters=it+1, rel=float(rel))
    if return_fields:
        out.update(r=r_c, th=th_c, Br=Br, Bth=Bth, Bmag=Bmag, M_r=M_r,
                   is_mag=g['is_mag'], is_iron=g['is_iron'])
    return out

if __name__ == '__main__':
    t0=time.time()
    for cfg,s in [('series',1.0),('parallel',0.5),('delta',1/np.sqrt(3))]:
        for th in [0,15,30]:
            for I in [30,60,120]:
                r=solve_case(th,s,NR=60,NT=120,I_phase=I)
                print(cfg,'th=%3d I=%3d T=%7.2f Tm=%7.2f Bgap=%.3f it=%d rel=%.1e'%(
                    th,I,r['T_lorentz'],r['T_maxwell'],r['Bgap'],r['iters'],r['rel']), flush=True)
    print('total %.1f s'%(time.time()-t0))
