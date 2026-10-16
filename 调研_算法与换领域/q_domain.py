# -*- coding: utf-8 -*-
"""B 组: 候选领域 x (ML代理密度, 相似律破缺类研究)"""
from oa import run

Q = [
 # ---- 相变 / 储能 ----
 ("pcm_ml",        '"phase change material" AND ("machine learning" OR "neural network" OR "surrogate")', None),
 ("stefan_ml",     '"Stefan problem" AND ("machine learning" OR "neural network" OR "surrogate")', None),
 ("solidif_ml",    '"solidification" AND ("machine learning" OR "neural network")', None),
 # ---- 多孔介质 ----
 ("porous_conv_ml",'"porous media" AND ("natural convection" OR "mixed convection") AND ("machine learning" OR "neural network" OR "surrogate")', None),
 ("darcy_break",   '"non-Darcy" AND ("machine learning" OR "neural network")', None),
 ("forchheimer_ml",'"Forchheimer" AND ("machine learning" OR "neural network")', None),
 # ---- 燃烧 ----
 ("combust_ml",    '"combustion" AND "surrogate" AND ("neural network" OR "chemistry")', None),
 ("flame_ml",      '"flame" AND "machine learning" AND ("LES" OR "large eddy simulation")', None),
 # ---- 电池 / 电化学 ----
 ("batt_ml",       '"lithium-ion battery" AND ("surrogate" OR "machine learning") AND ("thermal" OR "electrochemical")', None),
 ("batt_peukert",  '"Peukert"', None),
 ("fuelcell_ml",   '"fuel cell" AND ("surrogate" OR "neural network")', None),
 # ---- 固体力学 / 断裂 ----
 ("fracture_ml",   '"fracture" AND ("machine learning" OR "neural network") AND "surrogate"', None),
 ("sizeeffect_ml", '"size effect" AND ("machine learning" OR "neural network") AND ("fracture" OR "strength")', None),
 ("hphenvelope",   '"Hall-Petch"', None),
 # ---- 材料微结构 / 均匀化 ----
 ("micro_homog",   '"microstructure" AND "homogenization" AND ("machine learning" OR "neural network")', None),
 ("crystal_plast", '"crystal plasticity" AND ("machine learning" OR "neural network")', None),
 # ---- 地下 / 地质 ----
 ("subsurface_ml", '"subsurface flow" AND ("machine learning" OR "surrogate" OR "neural operator")', None),
 ("co2_ml",        '"CO2 storage" AND ("machine learning" OR "surrogate" OR "neural operator")', None),
 # ---- 电磁 / 光子 ----
 ("photonic_ml",   '"photonics" AND ("neural network" OR "surrogate" OR "deep learning")', None),
 ("em_surrogate",  '"electromagnetic" AND "surrogate model"', None),
 # ---- 等离子体 / MHD ----
 ("plasma_ml",     '"plasma" AND ("surrogate" OR "neural operator")', None),
 ("mhd_ml",        '"magnetohydrodynamic" AND ("machine learning" OR "neural network")', None),
 # ---- 声学 / 波 ----
 ("acoustics_ml",  '"acoustics" AND ("surrogate" OR "neural operator" OR "neural network")', None),
 ("wave_scat_ml",  '"wave scattering" AND ("neural operator" OR "machine learning")', None),
 ("seismic_ml",    '"seismic" AND ("neural operator" OR "surrogate")', None),
 # ---- 半导体 / 电子散热 ----
 ("tcad_ml",       '"TCAD" AND ("machine learning" OR "neural network")', None),
 ("chip_thermal",  '"chip" AND "thermal" AND ("surrogate" OR "machine learning")', None),
 # ---- 湍流闭合 ----
 ("rans_ml",       '"RANS" AND ("machine learning" OR "data-driven") AND "closure"', None),
 ("les_closure",   '"large eddy simulation" AND "subgrid" AND ("machine learning" OR "neural network")', None),
 ("turb_scaling",  '"turbulence" AND ("scaling law" OR "similarity") AND ("machine learning" OR "neural network")', None),
 # ---- 相场 ----
 ("phasefield_ml", '"phase field" AND ("machine learning" OR "neural network" OR "surrogate")', None),
 # ---- 微流体 / 血流 ----
 ("microfluid_ml", '"microfluidics" AND ("machine learning" OR "neural network")', None),
 ("hemo_ml",       '"hemodynamics" AND ("surrogate" OR "machine learning")', None),
 # ---- 海洋 / 波 ----
 ("oceanwave_ml",  '"ocean waves" AND ("machine learning" OR "neural network")', None),
 ("wavebreak_ml",  '"wave breaking" AND ("machine learning" OR "neural network")', None),
 # ---- 多相流 ----
 ("multiphase_ml", '"multiphase flow" AND ("machine learning" OR "neural network" OR "surrogate")', None),
 # ---- 稀薄气体 / 微纳传热 ----
 ("rarefied_ml",   '"rarefied gas" AND ("machine learning" OR "neural network")', None),
 ("knudsen_ml",    '"Knudsen number" AND ("machine learning" OR "neural network")', None),
 ("nonfourier_ml", '"non-Fourier" AND ("heat conduction" OR "thermal") AND ("machine learning" OR "neural network")', None),
 ("phonon_ml",     '"phonon transport" AND ("machine learning" OR "neural network")', None),
 ("bte_ml",        '"Boltzmann transport equation" AND ("machine learning" OR "neural network")', None),
 # ---- 气动 / 润滑 ----
 ("airfoil_ml",    '"airfoil" AND ("neural operator" OR "surrogate")', None),
 ("ehl_ml",        '"elastohydrodynamic" AND ("machine learning" OR "neural network")', None),
 # ---- 增材制造 ----
 ("am_meltpool",   '"additive manufacturing" AND "melt pool" AND ("machine learning" OR "neural network")', None),
 # ---- 换热器 / 工程传热 ----
 ("heatexch_ml",   '"heat exchanger" AND ("surrogate" OR "machine learning")', None),
 ("nanofluid_ml",  '"nanofluid" AND ("machine learning" OR "neural network")', None),
 ("marangoni_ml",  '"Marangoni" AND ("machine learning" OR "neural network")', None),
 ("nonnewton_ml",  '"non-Newtonian" AND ("machine learning" OR "surrogate")', None),
 # ---- 通用: 经验关联式被 ML 修正 ----
 ("corr_emprical", '"empirical correlation" AND ("machine learning" OR "neural network")', None),
 ("model_discrep", '"model discrepancy" AND ("machine learning" OR "Gaussian process")', None),
 ("closure_data",  '"data-driven closure"', None),
 ("simplaw_break", '"similarity law" AND ("breakdown" OR "breaking" OR "departure")', None),
]

if __name__ == "__main__":
    run(Q, "domain.json", per_page=5)
