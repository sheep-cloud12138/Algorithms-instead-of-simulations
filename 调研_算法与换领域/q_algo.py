# -*- coding: utf-8 -*-
"""A 组: ML 算法族 × PDE/仿真代理  文献密度"""
from oa import run

Q = [
 # --- 主流算子学习 ---
 ("fno",        '"Fourier neural operator"', None),
 ("deeponet",   '"DeepONet"', None),
 ("neural_op",  '"neural operator"', None),
 ("op_learning",'"operator learning"', None),
 # --- PINN 系 ---
 ("pinn",       '"physics-informed neural network"', None),
 ("pinn_2025",  '"physics-informed neural network"', 2025),
 # --- 图/网格 ---
 ("gnn_mesh",   '"graph neural network" AND "mesh" AND "simulation"', None),
 ("meshgraphnet",'"MeshGraphNet"', None),
 # --- Transformer / attention ---
 ("transformer_pde", '"transformer" AND "partial differential equation" AND "surrogate"', None),
 ("gnot",       '"Galerkin transformer" OR "gnot"', None),
 # --- 新兴架构 ---
 ("kan",        '"Kolmogorov-Arnold network"', None),
 ("kan_pde",    '"Kolmogorov-Arnold network" AND ("partial differential equation" OR "PDE")', None),
 ("mamba_pde",  '"Mamba" AND ("partial differential equation" OR "PDE")', None),
 ("diffusion_pde",'"diffusion model" AND "partial differential equation" AND "surrogate"', None),
 ("neural_ode", '"neural ordinary differential equation"', None),
 ("koopman",    '"Koopman" AND ("machine learning" OR "deep learning")', None),
 ("hyena_ssm",  '"state space model" AND "partial differential equation"', None),
 # --- 基座模型 ---
 ("fm_sci",     '"foundation model" AND ("scientific machine learning" OR "PDE" OR "partial differential equation")', None),
 ("fm_2025",    '"foundation model" AND ("scientific machine learning" OR "PDE" OR "partial differential equation")', 2025),
 ("llm_pde",    '"large language model" AND ("partial differential equation" OR "PDE" OR "simulation")', None),
 ("pos_emb",    '"PDE" AND "pre-trained" AND "neural operator"', None),
 # --- 物理先验注入 ---
 ("equiv_pde",  '"equivariant" AND ("neural operator" OR "operator learning")', None),
 ("equiv_nn",   '"equivariant neural network"', None),
 ("dim_ml",     '"dimensional analysis" AND ("machine learning" OR "neural network")', None),
 ("buckingham_ml",'"Buckingham" AND ("machine learning" OR "neural network")', None),
 ("symplectic", '"Hamiltonian neural network" OR "symplectic neural network"', None),
 ("structure_preserving",'"structure-preserving" AND ("neural network" OR "neural operator")', None),
 # --- 多保真 / 残差修正 ---
 ("multifidelity",'"multi-fidelity" AND ("neural network" OR "surrogate" OR "machine learning")', None),
 ("lowfi_correction",'"low-fidelity" AND "high-fidelity" AND ("correction" OR "residual")', None),
 ("transfer_sim",'"transfer learning" AND "surrogate" AND ("simulation" OR "CFD")', None),
 # --- 主动学习/采样 ---
 ("active_learn",  '"active learning" AND ("simulation" OR "CFD") AND ("surrogate" OR "neural")', None),
 ("bayes_opt_sim", '"Bayesian optimization" AND ("simulation" OR "CFD")', None),
 # --- 外推 / OOD ---
 ("extrap_op",  '"extrapolation" AND ("neural operator" OR "operator learning")', None),
 ("ood_pde",    '"out-of-distribution" AND ("neural operator" OR "operator learning" OR "surrogate model")', None),
 ("generalize_adv",'"generalization" AND "neural operator" AND "geometry"', None),
 # --- 不确定度 ---
 ("uq_surr",    '"uncertainty quantification" AND "surrogate" AND "simulation"', None),
 ("conformal",  '"conformal prediction" AND ("surrogate" OR "simulation" OR "PDE")', None),
 # --- 工业 ---
 ("cae_dl",     '"CAE" AND ("deep learning" OR "neural network")', None),
 ("digital_twin",'"digital twin" AND ("surrogate" OR "reduced order")', None),
 ("rl_design",  '"reinforcement learning" AND ("design optimization" OR "flow control")', None),
]

if __name__ == "__main__":
    run(Q, "algo.json", per_page=6)
