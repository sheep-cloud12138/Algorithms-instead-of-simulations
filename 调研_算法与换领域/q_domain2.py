from oa import run
Q = [
 ("subsurface_ml", '"subsurface flow" AND ("machine learning" OR "surrogate" OR "neural operator")', None),
 ("co2_ml",        '"CO2 storage" AND ("machine learning" OR "surrogate" OR "neural operator")', None),
 ("photonic_ml",   '"photonics" AND ("neural network" OR "surrogate" OR "deep learning")', None),
 ("em_surrogate",  '"electromagnetic" AND "surrogate model"', None),
 ("nonnewton_ml",  '"non-Newtonian" AND ("machine learning" OR "surrogate")', None),
 ("corr_empirical",'"empirical correlation" AND ("machine learning" OR "neural network")', None),
 ("model_discrep", '"model discrepancy" AND ("machine learning" OR "Gaussian process")', None),
 ("closure_data",  '"data-driven closure"', None),
 ("simlaw_break",  '"similarity law" AND ("breakdown" OR "breaking" OR "departure")', None),
 ("scaling_break", '"scaling law" AND ("breakdown" OR "breaks down")', None),
 ("mixture_conv",  '"mixed convection" AND ("machine learning" OR "neural network" OR "surrogate")', None),
 ("varprop_nc",    '"variable properties" AND ("natural convection" OR "Rayleigh-Benard")', None),
]
run(Q, "domain2.json", per_page=5)
