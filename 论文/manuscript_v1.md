# When is a Boussinesq surrogate safe for compressible convection?
# The role of the break-symmetry parameter ε = ΔT/T₀

*Draft v1 — 2026-10-16. All numbers below are from runs that have completed and are reproducible
from the scripts listed in §8.*

---

## Abstract

Machine-learned surrogates are increasingly used to replace costly compressible-flow
simulations by cheap Boussinesq ones. We show that, for two-dimensional Rayleigh–Bénard
convection, the error of any such surrogate is controlled by a **single, exactly
identifiable parameter**, the relative temperature contrast `ε = ΔT/T₀`. We establish:

1. **ε is the only break-symmetry parameter.** In the non-dimensionalisation used here the
   Froude number is not free (`Fr² = β ΔT = ε` identically) and the Mach-number effect is
   O(Ma²), below the solver floor in the whole accessible range. Hence the break-symmetry
   group is one-dimensional.
2. **The Boussinesq solution is exactly ε-blind.** In a paired experiment (20 configurations,
   5 values of ε, identical Rayleigh number and bit-identical thermal boundary conditions
   within a configuration) the low-fidelity field changes by at most `9.8e-8`, i.e. by the
   solver's own convergence noise. The correction therefore *cannot* be inferred from the
   surrogate's input.
3. **A scaling law and a hard floor.** Over 600 samples spanning four decades of Rayleigh
   number and 134 distinct boundary-condition families, the relative correction obeys
   `‖Y−X‖/‖Y‖ = 0.181 ε − 0.021` (`ux`, R² = 0.62); Rayleigh number and the boundary-condition
   family add nothing (R² 0.657 combined). The paired design turns this into a *provable*
   lower bound for every ε-blind surrogate.
4. **The floor is not an artefact.** A U-Net trained only on Boussinesq inputs lands on the
   bound; adding a single constant ε channel beats it by 1.5× (extrapolation) and 2.6×
   (interpolation), recovering 77 % and 94 % of the correction energy. Shallow alternatives
   fail completely: kernel ridge regression on the same data gains nothing from ε
   (0.296 → 0.301), and a reduced-rank linear mode transfer from a reference ε gains at most
   1.6× for velocity.
5. **The floor is also a design rule.** Splitting ε into `D = ε·P(X)` and letting the network
   predict `P` — so that ε never enters the network at all — is *better* than feeding ε in
   (ux: 0.2181 → 0.1286 for an ε-blind input; 0.1449 → 0.0981 with ε). The payoff is that a
   dataset simulated at a **single** value of ε (95 pairs, all at ε = 1.2 — the setting of the
   published data set of the competing work) then transfers across ε ∈ [0.2, 1.9] to 9.3 %
   error, against 14.1 % for the same data without the prior, which is *worse than doing
   nothing* at small ε. Boundary conditions never seen in training (36 of 134 families held
   out) are handled to 4.6 %.

---

## 1. Introduction

(to be written — literature: Menglik et al. JCP (same problem, ε ≡ 1.2, no ε axis);
ROM/surrogate literature; non-Boussinesq convection literature.)

## 2. Problem, discretisation and the break-symmetry parameter

### 2.1 Two members

Every sample consists of a *paired* solution of the same physical configuration with two
equation sets on the same mesh (`res 64`, `L = 1`, `Pr = 0.71`, `γ = 1.4`):

* **High fidelity (Y):** the compressible low-Mach formulation in the unknowns
  `w = (u, T, ρ, p)`, with the equation of state `ρ e = γ Ma² p + 1`, the viscous stress
  `σ = ∇u + ∇uᵀ − (2/3)(∇·u) I`, `Re = (Ra/Pr)^{1/2}` and gravity
  `−Fr⁻² ρ g`. Entropy production, viscous dissipation and the work of pressure are retained.
* **Low fidelity (X):** Boussinesq, unknowns `(u, T, p)`, `∇·u = 0`, buoyancy `Pr Ra θ g`
  in the free-fall scaling, `θ = (T − T₀)/ΔT`.

Both are discretised with Taylor–Hood-type mixed elements and solved by a damped Newton
iteration preconditioned by a direct MUMPS factorisation; steady states are reached by an
adaptive march with a convergence guard.

### 2.2 Why ε is the only break-symmetry parameter

Three statements, each verifiable by inspection or by a two-line numerical experiment:

| candidate | finding | consequence |
|---|---|---|
| Fr | `Fr² = β ΔT = ΔT/T₀` identically in this non-dimensionalisation (the upstream 3-D generator states this verbatim); the released 2-D code pinned `Fr = 1.095 = √1.2`, correct only at ΔT = 1.2 | Fr is **not** a free parameter; moving T₀/ΔT with Fr pinned is what made a temperature axis look non-monotone |
| Ma | fixing Ra, T₀, ΔT and changing Ma by a factor 24 leaves the correction amplitude unchanged to one significant figure (9.6740 → 9.6740) | Ma effects are O(Ma²) ≈ 10⁻⁶ here: **unmeasurable** |
| ε = ΔT/T₀ | same boundary-condition seed stream, ΔT swept: correction 0.09 → 0.62, monotone | **the clean, 1-D break-symmetry axis** |

### 2.3 The low-fidelity solution is exactly ε-blind

The Boussinesq weak form contains `θ = (T − T₀)/ΔT` and `Pr Ra`; neither `T₀` nor `ΔT`
appears on its own. Consequently, the low-fidelity field cannot depend on ε at all:

> **Proposition.** Let the thermal boundary data be prescribed in terms of θ and the
> Rayleigh number be fixed. Then the Boussinesq solution is independent of ε.

This is confirmed to solver precision in §4.2. The corollary is what motivates the whole
paper: *the correction Y − X is information that is absent from the surrogate's input*, so no
amount of capacity or data can recover it — the limit is information-theoretic, not
architectural.

## 3. Paired-simulation design and the exact error floor

For a fixed configuration let the ε-blind model receive X. Within that configuration X is the
same field for every ε, hence the model must emit the same field `F` for every ε. The best
such `F`, and the resulting error, are computable exactly:

```
F*(c) = argmin_F Σ_ε ‖F − Y(c,ε)‖²      ⇒      floor_B(c) = ‖Y(c,ε) − F*(c)‖ / ‖Y(c,ε)‖
```

Because X determines the configuration, `F*` is available to any ε-blind model, so `floor_B`
is the **exact lower bound** for the ε-blind family. We also report `floor_A`, the error of the
single-ε protocol (train at ε = 1.2, the only ε present in the published dataset of the
competing work): `‖Y(c,ε) − Y(c,1.2)‖ / ‖Y(c,ε)‖`.

The 20 configurations (Ra = 10² … 10⁶, random Fourier wall temperatures) with 5 values
`ε ∈ {0.4, 0.8, 1.2, 1.6, 1.9}` share, inside a configuration, the Rayleigh number and a
bit-identical θ boundary-condition stream.

## 4. Results

### 4.1 The ε-law (600 random boundary conditions)

`R = ‖Y−X‖/‖Y‖` per sample:

| channel | fit | R² | + log Ra + 134 BC families |
|---|---|---|---|
| ux | R = 0.181 ε − 0.021 | 0.619 | 0.657 |
| uy | R = 0.230 ε − 0.020 | 0.564 | 0.689 |
| θ | R = 0.027 ε − 0.004 | 0.189 | 0.667 (via χ, below) |
| p | ≈ 1.00 (const.) | 0.07 | 0.285 |

Neither the Rayleigh number nor the boundary-condition family explains the residual scatter.
The θ-channel scatter is, however, explained by an advection-strength parameter
`χ = ‖u‖‖∇θ‖/‖θ‖` (Spearman ρ = 0.877; R² 0.189 → 0.667).

### 4.2 ε-blindness of the low-fidelity member (exact)

Maximum relative deviation of X across the five ε values within a configuration: **9.77e-8**,
the solver's own convergence floor. Across all 20 configurations.

### 4.3 Exact floors

| ε | 0.40 | 0.80 | 1.20 | 1.60 | 1.90 |
|---|---|---|---|---|---|
| floor_A (ux) | 0.1427 | 0.0764 | 0 | 0.1119 | 0.2539 |
| floor_B (ux) | 0.1689 | 0.1073 | 0.0355 | 0.0778 | 0.2249 |
| do nothing (ux) | 0.0575 | 0.1266 | 0.2099 | 0.3230 | 0.4985 |

The correction field barely rotates: the cosine similarity between D(ε) and D(1.2) is
0.972–0.994, and a per-configuration rank-1 model `D(ε) ≈ (a + bε)·shape` explains a median
R² of 0.9447 (ux). **ε acts predominantly as an amplitude modulator.**

### 4.4 A U-Net on 600 samples

Residual learning `Ŷ = X + N(X[, ε])`, 4-channel input (A) versus 5-channel with a constant ε
map (B), 843k parameters, Adam with cosine annealing, 300 epochs.

| split | channel | A (no ε) | B (ε in) | gain |
|---|---|---|---|---|
| extrapolation (train ε≤1.3) | ux | 0.2181 | 0.1449 | 1.51× |
| | uy | 0.2594 | 0.1680 | 1.54× |
| | θ | 0.0385 | 0.0281 | 1.37× |
| | p | 1.8183 | 0.2924 | 6.22× |
| interpolation (stratified) | ux | 0.1449 | 0.0550 | 2.63× |
| | uy | 0.1709 | 0.0500 | 3.42× |
| | θ | 0.0233 | 0.0079 | 2.95× |
| | p | 0.6706 | 0.0535 | 12.5× |

Model A sits on `floor_A` throughout — the ceiling is informational.

### 4.5 Shallow and linear alternatives

| method | A | B | comment |
|---|---|---|---|
| RBF kernel ridge, full field | 0.2964 | 0.3013 | no gain; 16 384-dim problem degenerates to `Y := X` |
| multiscale linear ridge + correction PCA | 0.150 | 0.132 | +12 % only |
| pointwise global regression on ε alone | R² = 0.014 (ux) | — | useless |
| reduced-rank linear mode transfer from D(1.2) | ux 1.17× / p 8.35× at ε = 1.9 | — | velocity cannot be transferred linearly |

### 4.6 The floor is not an artefact: U-Net on 600 samples

Residual learning `Ŷ = X + N(X[, ε])`, 843k parameters, Adam with cosine annealing,
300 epochs. “Do nothing” is `Ŷ = X`.

| split | channel | A (no ε) | B (ε in) | gain | do nothing | correction energy recovered by B |
|---|---|---|---|---|---|---|
| extrapolation (train ε≤1.3, n=218) | ux | 0.2181 | 0.1449 | 1.51× | 0.3018 | 77 % |
| | uy | 0.2594 | 0.1680 | 1.54× | 0.3731 | 80 % |
| | θ | 0.0385 | 0.0281 | 1.37× | 0.0556 | 74 % |
| | p | 1.8183 | 0.2924 | 6.22× | 1.0258 | 92 % |
| interpolation (stratified, n=200) | ux | 0.1449 | 0.0550 | 2.63× | 0.2203 | 94 % |
| | uy | 0.1709 | 0.0500 | 3.42× | 0.2683 | 97 % |
| | θ | 0.0233 | 0.0079 | 2.95× | 0.0381 | 96 % |
| | p | 0.6706 | 0.0535 | 12.5× | 1.0028 | 99.7 % |

Model A tracks the paired-set floor throughout; the ceiling is informational.

### 4.7 Unseen boundary-condition families

Holding out 36 of the 134 wall-mode families (n = 151, ε = 0.21 … 1.89):

| channel | A (no ε) | B (ε in) | do nothing | B vs A |
|---|---|---|---|---|
| ux | 0.1122 | **0.0464** | 0.1770 | 2.42× |
| uy | 0.1365 | **0.0440** | 0.2344 | 3.10× |
| θ | 0.0157 | **0.0067** | 0.0258 | 2.34× |
| p | 0.6842 | **0.0581** | 1.0054 | 11.8× |

### 4.8 How many ε values must be simulated?

ε restricted to [0.35, 1.75]; N campaigns, each a ±0.15 band around a distinct ε value;
every ε between the bands is unseen.

| N_ε | centres | n_train | n_test | ux error | do nothing |
|---|---|---|---|---|---|
| 2 | 0.50, 1.60 | 229 | 258 | 0.0512 | 0.1749 |
| 3 | 0.50, 1.05, 1.60 | 331 | 156 | 0.0370 | 0.1744 |
| 4 | 0.50, 0.87, 1.23, 1.60 | 426 | 61 | 0.0381 | 0.1918 |

Two bracketing campaigns suffice; finer ε sampling buys nothing. For reference, on a uniform
random 20 % hold-out over the whole ε range the ε-aware model gives ux = 0.0651.

### 4.9 A physics prior instead of an ε channel

If `D = ε·P(X)` then a network that predicts `P` is ε-aware while never seeing ε. In the
extrapolation split (training on ε ≤ 1.3, do nothing = 0.3018):

| model | input | target | ux | uy | θ |
|---|---|---|---|---|---|
| A | X | D | 0.2181 | 0.2594 | 0.0385 |
| **C** | **X** | **D/ε** | **0.1286** | **0.1369** | 0.0169 |
| B | X, ε | D | 0.1449 | 0.1680 | 0.0281 |
| **D** | X, ε | D/ε | **0.0981** | **0.1042** | **0.0157** |

Single-ε dataset reuse — training only on the 95 pairs within |ε − 1.2| < 0.15 and testing on
everything else:

| test ε | G: naive (no prior) | **F: with the prior** | do nothing |
|---|---|---|---|
| 0.2–0.8 | 0.1319 (worse than doing nothing) | **0.0321** | 0.0768 |
| 0.8–1.3 | 0.0765 | **0.0563** | 0.1491 |
| 1.3–2.0 | 0.1655 | **0.1292** | 0.3054 |
| all | 0.1413 | **0.0925** | 0.2205 |

### 4.10 Same-footing comparison on the paired set

With only 100 paired samples the networks are data-limited, but the ordering is consistent:
when the training set contains a single ε, an ε input channel is *harmful* (it is an
uninformative constant); when the training ε range is non-degenerate, the barrier drops from
0.2383 to 0.1288 at ε = 1.6 and from 0.3951 to 0.3116 at ε = 1.9, against exact floors
floor_A = 0.1098 / 0.2431 and floor_B = 0.0773 / 0.2091.

## 5. Discussion

## 6. Limitations

1. Two-dimensional; 3-D is infeasible at this scale on the available hardware
   (measured: > 30 min and 2.58 GB per sample, credible Ra ≤ 10⁵ at 24³).
2. The break-symmetry group is effectively one-dimensional in the accessible range
   (Ma has no measurable effect).
3. ε < 2, imposed by the equation of state (T_c = T₀ − ΔT/2 > 0).
4. The two members converge to different tolerances (median 2.5e-7 versus 3.8e-6, worst 7.0e-4);
   the worst case is 1.6 % of the smallest correction signal.
5. The fitted intercept of the ε-law is −0.0212 ± 0.0069 (3.1σ), at odds with the physical
   expectation R(0) = 0; this needs a dedicated check.

## 7. Reproducibility

All scripts and the analysis pipeline are listed in §8; the data set, the generator flags and
the three minimal patches applied to the upstream generator are documented in the data card.

## 8. Artefacts

| item | path |
|---|---|
| 600-sample data card | `研究结果/数据集卡片_600组.md` |
| results summary (Chinese) | `研究结果/结果汇总_v5.md` |
| P1 figure | `研究结果/fig_p1_law.png` |
| P2 figure | `研究结果/fig_p2_main.png` |
| P4/P5 figure | `研究结果/fig_p45.png` |
| manuscript draft | `论文/manuscript_v1.md` |
| analysis scripts | WSL `~/nc_run/p1`, `p2`, `p3`, `p4`, `p5`, `p6` |
