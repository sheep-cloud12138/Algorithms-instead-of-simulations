"""Timing of one direct linear solve."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_direct_linear as M
import ufl
from dolfinx import fem

BASE = os.path.expanduser("~/nc_run/paired/dataset")
case, eps = "c16_Ra1e6_1100", 1.2
d = os.path.join(BASE, "%s_e%.1f" % (case, eps), "sample_000000")
meta = json.load(open(os.path.join(d, "meta.json")))
Ra = float(meta["Ra"]); res = int(meta["res"]); L = float(meta["length"]); Ma = float(meta["Ma"])
t0 = time.time(); msh = M.G.create_mesh(L, res); W = M.G.space_comp(msh); t_mesh = time.time() - t0
t0 = time.time(); bcs = M.make_bcs(msh, W, meta, L); wx = M.build_X(msh, W, d, eps); t_load = time.time() - t0
F = M.residual(msh, W, wx, Ra, eps, Ma)
Lf = -ufl.replace(F, {wx: wx}); Jf = ufl.replace(ufl.derivative(F, wx), {wx: wx})
t0 = time.time(); A = fem.petsc.assemble_matrix(fem.form(Jf), bcs=bcs); A.assemble(); t_asm = time.time() - t0
b = fem.petsc.assemble_vector(fem.form(Lf)); fem.petsc.apply_lifting(b, [fem.form(Jf)], [bcs])
b.ghostUpdate(addv=__import__("petsc4py").PETSc.InsertMode.ADD, mode=__import__("petsc4py").PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, bcs)
from petsc4py import PETSc
t0 = time.time()
ksp = PETSc.KSP().create(msh.comm); ksp.setOperators(A); ksp.setType("preonly")
pc = ksp.getPC(); pc.setType("lu"); pc.setFactorSolverType("mumps")
ksp.setUp(); t_setup = time.time() - t0
x = A.createVecRight(); t0 = time.time(); ksp.solve(b, x); t_solve = time.time() - t0
print("mesh %.2fs load %.2fs assemble %.2fs factor_setup %.2fs solve %.2fs total %.2fs" %
      (t_mesh, t_load, t_asm, t_setup, t_solve, t_mesh + t_load + t_asm + t_setup + t_solve))
print("dofs", W.dofmap.index_map.size_local * W.dofmap.index_map_bs)
