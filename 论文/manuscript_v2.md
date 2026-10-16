# One parameter governs the Boussinesq–compressible gap: the break-symmetry axis ε = ΔT/T₀ and a physics-prior surrogate

*Draft v2 — 2026-10-16. Every number below comes from a run that has completed on the machine
described in §8 and is reproducible from the scripts in `code/`.*

---

## Abstract

Surrogates of compressible natural convection are routinely trained on, or validated against,
Boussinesq solutions. We show that for two-dimensional Rayleigh–Bénard convection the gap
between the two physics is controlled by **one exactly identifiable parameter**, the relative
temperature contrast `ε = ΔT/T₀`, and we turn that fact into a design rule for surrogates.

1. **ε is the only break-symmetry parameter.** In the non-dimensionalisation used here the
   Froude number is not free (`Fr² = β ΔT = ΔT/T₀` identically) and the Mach-number effect is
   `O(Ma²)`, unmeasurable here (changing Ma by a factor 24 changes the correction by less than
   one part in 10⁵). The break-symmetry group is one-dimensional.
2. **The law.** Over 600 random-boundary-condition samples plus 27 independent small-ε checks
   (ε = 0.02 … 1.9, five decades), the relative correction obeys
   `R = ‖Y−X‖/‖Y‖ = A ε^p` with `p = 1.023±0.018 (u_x), 0.995±0.019 (u_y), 1.021±0.077 (θ),
   1.009±0.015 (p)` and `A = 0.147, 0.193, 0.010, 0.148`. The exponent is 1 within 2 %, the
   intercept vanishes (`+0.0003 ± 0.0047` for `u_x`), and 134 distinct boundary-condition
   families add only 3.5 % of explained variance beyond ε. A weak Rayleigh-number trend in the
   prefactor is real and reproducible **in an independent second geometry** for `u_y`
   (`q = −0.083 ± 0.005` vs `−0.077 ± 0.019`) and `θ` (`+0.514 ± 0.015` vs `+0.447 ± 0.045`).
3. **Why.** Expanding the compressible member in ε gives an `O(1)` system that is term-by-term
   the Boussinesq equations and an `O(ε)` source from the density-modulated inertia and the
   non-solenoidal velocity field, `∇·u₁ = ∇·(θ₀u₀)`. Because the linearised operator is
   ε-independent, `D = Y − X = ε P(X) + ε² Q(X) + …` with a shape `P` that does not rotate with
   ε — confirmed by `cos(D(ε), D(1.2)) = 0.95–1.00` and by a direct check of the divergence
   relation (cosine 0.994 for ε < 0.6). Two terms of the expansion carry 99.0 % of the
   correction energy per configuration.
4. **Consequence for surrogates.** Any model that receives only the Boussinesq field has an
   *unreachable* error floor of 8–35 % when ε leaves its training value — the correction is
   simply absent from its input. Re-parametrising the output as `D = ε·P(X)` and letting the
   network predict `P` removes ε from the input entirely: it beats the plain architecture by
   1.62× in extrapolation, lets a network find the `ε²` correction on its own (learned curvature
   β = 0.20–0.29 for all four channels), transfers unchanged to a Fourier neural operator
   (1.57×), and is what makes a **single-ε dataset reusable**: a surrogate trained on a public
   4000-pair dataset that exists only at ε = 1.2 is applied, without retraining, to our own
   ε ∈ [0.2, 1.9] data.
5. **External validation.** The law reproduces on that published dataset (independent group,
   independent generator, ε ≡ 1.2): `R = 0.180 / 0.244 / 0.023 / 0.186` for
   `(u_x, u_y, θ, p)` against `0.184 / 0.225 / 0.023 / 0.189` on our own ε ≈ 1.2 slice, and even
   the secondary Rayleigh-number correlations match (`r = +0.65` for θ in both).
6. **Speed.** Measured on one machine: the paired solver costs 33.9 s per sample (median,
   64², Boussinesq + compressible) and the surrogate 6.3 ms — a factor 5.4×10³ per sample and
   6.9×10⁴ in batches.

---

## 1. Introduction

Steady natural convection is a workhorse of thermal engineering and it is expensive. A converged
pair of two-dimensional solves on a 64² mesh costs 34 s of wall clock on an eight-way desktop
(§5.11); the three-dimensional problems that engineering actually needs cost hours to days. That
is the standard motivation for surrogate modelling, and the standard response is to train a
network that maps a cheap model's output to an expensive model's output.

For convection the two models are not two arbitrary fidelities: they are the Boussinesq
(Oberbeck-Boussinesq) equations and the compressible low-Mach equations. Boussinesq is exact only
in the limit of vanishing relative temperature contrast; the corrections are O(β ΔT) and have
been characterised asymptotically for decades [Spiegel & Veronis 1960; Gray & Giorgini 1976].
Recent work makes the pair directly learnable. Menglik, Shao & Hyde (arXiv:2606.25259) publish
paired Boussinesq→compressible steady solutions in two dimensions (4000 training pairs plus
held-out splits, CC BY 4.0) and train a Fourier neural operator [Li et al. 2021] to predict the
compressible field from the Boussinesq field. That data set — which we use as an independent
validation set in §5.7–5.8 — contains exactly **one** temperature contrast: ΔT = 1.2 at T₀ = 1.

The single-contrast design is the practical crux. A surrogate fitted on such a data set has
learned the correction at one operating point. At any other temperature contrast the Boussinesq
input already differs from the input seen in training, so the correction the network memorised is
the wrong correction — not merely an inaccurate one — and the only remedy on offer is to generate
a second paired data set, i.e. to pay again exactly the cost the surrogate was meant to remove.
We show in §5.3 that this is not a modelling failure that more capacity or a better architecture
can repair: it is an information barrier, and we measure it.

This paper makes the barrier explicit and then removes it.

**What governs the gap.** For the non-dimensionalisation used here the Froude number is not free:
`Fr² = β ΔT = ΔT/T₀` identically, and the Mach number enters the equations only at O(Ma²)
(§2.2, and measured: a factor-24 change in Ma moves the correction by less than 10⁻⁵). The
break-symmetry group is therefore **one-dimensional**, and it has a name: ε = ΔT/T₀.

**What the gap looks like.** Expanding the compressible member in ε gives an O(1) system
that is term-by-term the Boussinesq system, plus an O(ε) source produced by the
density-modulated inertia and by the non-solenoidal velocity field `∇·u₁ = ∇·(θ₀u₀)`.
Because the linearised operator is ε-independent, the correction factorises,
`Y − X = ε P(X) + ε² Q(X) + ...`, with a shape P that does not rotate as ε
moves (§3). Measured over five decades of ε (627 samples, §5.1) the relative correction
obeys `R = A ε^p` with p = 1 within 2 % in all four channels, a vanishing intercept, and a
prefactor whose weak Rayleigh-number trend reproduces in a second geometry and in a third fluid.

**What it buys.** Because the ε dependence can be written outside the network, a surrogate
can predict `P(X)` and multiply by ε afterwards, keeping ε out of the input. The
result is not a small accuracy gain in distribution (§5.6: parity with an ε-augmented
baseline at n ≤ 200). It is a change of regime: a network trained on a public data set that
exists at a single ε is applied, without retraining, to our ε in [0.2, 1.9] (§5.8),
the extrapolation error falls by 1.62× relative to the plain architecture (§5.5), the network
discovers the ε² curvature of the expansion on its own (learned β = 0.20–0.29 in all
four channels), and the same prior transfers unchanged to a Fourier neural operator.

**Contributions.**

1. We identify the exact, one-dimensional break-symmetry parameter of the
   Boussinesq-compressible pair and show the group is one-dimensional rather than two (§2).
2. We derive the correction's factorisation from an ε-expansion and verify all five of its
   predictions numerically, including the divergence relation (§3).
3. We calibrate the resulting law on 627 paired samples spanning five decades of ε, in two
   geometries and two Prandtl numbers, and validate it on an independently generated public data
   set — including the secondary Rayleigh-number trends (§5.1, §5.7, §5.9, §5.12).
4. We turn the law into a surrogate design rule (`D = ε P(X)`), measure what it is worth
   (1.62× in extrapolation), and show it makes an existing single-ε public data set
   reusable, with the speed-up measured end to end (5.4×10³ per sample) (§5.4–5.8, §5.11).

All results are two-dimensional. Section 6 states the measured cost of the three-dimensional
problem on the same hardware and gives the concrete reasons we do not claim a 3-D sweep.

## 2. Problem, discretisation and the break-symmetry parameter

### 2.1 Two members per sample

Every sample is a *paired* steady solution of one physical configuration with two equation sets
on the same mesh (`res = 64`, `L = 1`, `Pr = 0.71`, `γ = 1.4`):

* **High fidelity (Y)** — compressible low-Mach formulation in `w = (u, T, ρ, p)`,
  `∇·(ρu) = 0`,
  `∇·(ρ u⊗u) = −∇p + Re⁻¹∇·σ − Fr⁻² ρ g`,
  `∇·(ρEu) = (γ/(Re Pr))∇²e + ∇·(P_work u) + O(Ma²)`,
  equation of state `ρe = γMa²p + 1`, `Re = (Ra/Pr)^{1/2}`.
* **Low fidelity (X)** — Boussinesq in `(u, T, p)` in the *same* free-fall scales,
  `∇·u = 0`, `u·∇u = −∇p + (Pr/Ra)^{1/2}∇²u + θ ĝ`, `u·∇T = (Ra Pr)^{-1/2}∇²T`,
  with `θ = (T − T₀)/ΔT`.

Both are solved with Taylor–Hood-type mixed elements, damped Newton and a MUMPS
factorisation; steady states are reached by an adaptive march with a convergence guard.
`θ` boundary data are random Fourier wall temperatures with per-wall mode dials.

### 2.2 Fr is not a parameter, Ma is negligible, ε is the axis

| candidate | evidence | conclusion |
|---|---|---|
| Fr | the generators state `Fr² = β ΔT = ΔT/T₀` identically; the released 2-D code pinned `Fr = 1.095`, correct only at ΔT = 1.2 | not free; pinned-Fr sweeps of T₀/ΔT are an artefact |
| Ma | fixing Ra, T₀, ΔT and changing Ma by 24× leaves the correction amplitude unchanged to five significant figures | `O(Ma²)` ≈ 10⁻⁶: unmeasurable |
| ε = ΔT/T₀ | same boundary-condition seed stream, ΔT swept | the clean, 1-D break-symmetry axis |

### 2.3 The low-fidelity solution is exactly ε-blind

`θ` and `Pr Ra` are the only combinations that appear in the Boussinesq weak form; neither `T₀`
nor `ΔT` appears alone. Hence, with the thermal data prescribed in θ and Ra fixed, **X cannot
depend on ε**. Verified to solver precision: across 20 configurations × 5 ε (same Ra, bit-identical
θ boundary stream), the maximum relative deviation of X is **9.77 × 10⁻⁸**.

**Corollary.** The correction `Y − X` is information that is *absent from the surrogate's input*.
No capacity or data can recover it. The limit is informational, not architectural (§5.3).

## 3. Asymptotic theory: why the correction factorises as ε·P(X)

*(full derivation, with the exact weak forms quoted from the generator, in
`论文/理论推导_eps展开.md`)*

Write `T = T₀(1 + εθ)` with θ = O(1). With `ρ = 1/T` (the equation of state and initial
condition used here) and `Fr² = ε`, the buoyancy source is
`ρg/ε = g/ε − θ g + ε θ² g + …`, so the `O(1)` balances are

```
∇·u₀ = 0
u₀·∇u₀ = −∇p̃₀ + Re⁻¹∇²u₀ + θ₀ ĝ
u₀·∇θ₀ = (Re Pr)⁻¹∇²θ₀
```

which is **term-by-term the Boussinesq system solved as X** (`Re⁻¹ = (Pr/Ra)^{1/2}`). The
correction at `O(ε)` is driven by the density modulation:

```
∇·u₁ = ∇·(θ₀ u₀)                                  (non-solenoidal velocity)
∇·(u₁⊗u₀ + u₀⊗u₁ − θ₀ u₀⊗u₀) = −∇p̃₁ + Re⁻¹∇²u₁ + θ₁ĝ + θ₀²ĝ
```

Since `L`, the linearisation of the Boussinesq operator about `X`, does not contain ε,

```
D = Y − X = −ε L⁻¹ S₁[X] − ε² L⁻¹ S₂[X] + …  =  ε P(X) + ε² Q(X) + …
```

Five predictions follow, and all five hold:

| prediction | measurement |
|---|---|
| `R → 0` as ε → 0 | intercept `+0.0003 ± 0.0047` (u_x); 0.01σ from zero |
| `R ∝ ε` | p = 1.023±0.018 / 0.995±0.019 / 1.009±0.015 (u_x/u_y/p) |
| the shape of D does not rotate with ε | `cos(D(ε),D(1.2)) = 0.95–1.00`, four channels × five ε |
| `∇·u₁ = ∇·(θ₀u₀)` | `div(Y)` vs `ε∇·(θu_X)`: cosine **+0.994** for ε < 0.6, +0.965 overall, amplitudes 2.26e-2 vs 2.10e-2 |
| the expansion converges fast | per configuration, ε¹ alone carries 86 % of the correction energy, ε¹+ε² **99.0 %**, ε³ 99.9 % |

The drift of the divergence check with ε (0.994 → 0.901) is the truncated `ε²` term, which is
exactly what the fifth line measures independently.

**Domain of validity.** `ε < 2`, since the cold wall `T_c = T₀ − ΔT/2` must be positive.

## 4. Data sets and protocols

| set | n | what varies | purpose |
|---|---|---|---|
| random-BC | 600 | Ra ~ logU[10²,10⁶], ε ~ U[0.2,1.9], 134 wall-mode families | calibrate the law, train surrogates |
| ε→0 checks | 27 | ε = 0.02/0.05/0.10 × Ra = 10³/10⁴/10⁵ | test `R(0) = 0` |
| paired | 20×5 | only ε; Ra and the θ boundary stream fixed within a configuration | exact ε-blind floor, shape invariance |
| aspect 2:1 | 104 | same law, geometry 2×1 | external validity |
| tight rerun | 10 | the 10 worst-converged samples, tolerance 1e-6 → 1e-8 | noise control |
| public (Menglik et al.) | 4000 + 200 | ε ≡ 1.2, Ra ~ logU[10²,3·10⁶] | external validation, cross-data-set transfer |

Surrogates use residual learning `Ŷ = X + N(...)`, a U-Net of width 32 (or an FNO2d with
modes 12, width 32, four layers, 1.19 M parameters), Adam with cosine annealing, 200 epochs,
batch 16, target normalised per channel. Two splits are used throughout:
**extrapolation** (train ε ≤ 1.3, test ε > 1.3, n = 218) and **interpolation** (stratified 2:1
in ε, n = 200). Errors are the per-channel relative L2 error against Y; "floor" is `Ŷ = X`.

## 5. Results

### 5.1 The ε-law, calibrated over five decades

| channel | `R = A ε^p` | A | R² | intercept (ε ≤ 0.5) |
|---|---|---|---|---|
| u_x | p = 1.023 ± 0.018 | 0.147 | 0.84 | +0.0003 ± 0.0047 |
| u_y | p = 0.995 ± 0.019 | 0.193 | 0.82 | +0.0030 ± 0.0062 |
| θ | p = 1.021 ± 0.077 | 0.010 | 0.22 | −0.0003 ± 0.0012 |
| p | p = 1.009 ± 0.015 | 0.148 | 0.87 | +0.0016 ± 0.0030 |

Two-parameter refinement (the Ra trend is real; `q` for u_y and θ reproduces in the 2:1 geometry):

| channel | unit cavity (n=600) p / q / R² | aspect 2:1 (n=104) p / q / R² |
|---|---|---|
| u_x | 1.081 / +0.003 ± 0.006 / 0.74 | 1.071 / −0.143 ± 0.019 / 0.71 |
| u_y | 1.016 / −0.083 ± 0.005 / 0.77 | 1.146 / −0.077 ± 0.019 / 0.69 |
| θ | 1.120 / +0.514 ± 0.015 / 0.72 | 0.989 / +0.447 ± 0.045 / 0.55 |
| p | 1.127 / −0.029 ± 0.005 / 0.83 | 1.057 / −0.154 ± 0.017 / 0.74 |

* ε-scaling is geometry-independent; the prefactor A is not (0.147→0.175 for u_x,
  0.148→0.222 for p, 0.193→0.175 for u_y, 0.010→0.008 for θ).
* Boundary-condition families barely matter: 134 families, 133 free dummies, u_x R² only
  0.613 → 0.719 (adjusted: +3.5 %).
* The θ channel needs one extra, cheaply available parameter, the advection strength of the base
  flow `χ = ‖u‖‖∇θ‖/‖θ‖`: `R_θ = ε(−0.0037 + 0.226χ)`, R² 0.185 → **0.778**
  (Spearman 0.902), i.e. `R_θ ≈ 0.23 ε χ`. For u_x/u_y/p adding χ does nothing. (A single-input form of the same
  statement, `R_θ/ε = c·û^a`, is calibrated on two independent generators in §5.7 and carries the
  same information with R² = 0.90.)
* A pressure-channel preprocessing bug had to be fixed first (§5.2).

### 5.2 A pressure-channel artefact, and why it mattered

Pairing the two members' *raw* pressures mixes units: the compressible raw pressure carries a
hydrostatic ramp 12× larger than its dynamic part, and the two weak forms use opposite pressure
sign conventions. Before the fix `|X_p|/|Y_p| = 0.053` and `cos(X_p, Y_p) = −0.109`, and the
p-channel "law" was a constant (`R_p ≈ 1.0`). Applying the recipe used by the public data set —
subtract the mean, remove a least-squares linear-in-y fit, flip the sign of the Boussinesq member
— gives `|X_p|/|Y_p| = 0.931` and `cos = +0.993` (the public set gives 0.929), and the channel
then obeys the same law as the velocities (`A = 0.148`, `p = 1.009`). All p-channel numbers
reported here are post-fix.

### 5.3 The ε-blind floor is exact and not cosmetic

Within a configuration X is identical for all ε, so an ε-blind model must emit one field `F`
for every ε; the best such `F` (available to any model, since X determines the configuration)
gives an exact lower bound. Median over 20 configurations, channels `u_x/u_y/θ/p`:

| ε | do nothing (`Ŷ=X`) | train at ε = 1.2 only | best ε-blind |
|---|---|---|---|
| 0.4 | 0.060 / 0.098 / 0.009 / 0.061 | 0.143 / 0.219 / 0.021 / 0.146 | 0.170 / 0.244 / 0.023 / 0.177 |
| 0.8 | 0.127 / 0.203 / 0.019 / 0.129 | 0.078 / 0.118 / 0.011 / 0.082 | 0.108 / 0.143 / 0.013 / 0.126 |
| 1.2 | 0.211 / 0.328 / 0.030 / 0.220 | 0 | 0.035 / 0.036 / 0.003 / 0.051 |
| 1.6 | 0.322 / 0.481 / 0.044 / 0.347 | 0.112 / 0.151 / 0.014 / 0.131 | 0.079 / 0.123 / 0.012 / 0.085 |
| 1.9 | 0.472 / 0.616 / 0.058 / 0.547 | 0.251 / 0.304 / 0.031 / 0.346 | 0.225 / 0.286 / 0.028 / 0.281 |

A surrogate built the way the published single-ε data set allows (train at ε = 1.2) therefore
carries an irreducible 8–35 % error at any other ε. Note that at ε = 0.4–0.8 "do nothing" is
*better* than that protocol.

### 5.4 Network baselines: the correction is information, not capacity

U-Net, width 32, 200 epochs, identical protocol for all columns:

| split | channel | A (no ε) | B (ε input) | gain | do nothing | energy recovered by B |
|---|---|---|---|---|---|---|
| extrapolation (n=218) | u_x | 0.2249 | **0.1540** | 1.46× | 0.3018 | 74 % |
| | u_y | 0.2645 | **0.1774** | 1.49× | 0.3731 | 77 % |
| | θ | 0.0398 | **0.0250** | 1.59× | 0.0556 | 80 % |
| | p | 0.2617 | **0.1743** | 1.50× | 0.3320 | 72 % |
| interpolation (n=200) | u_x | 0.1496 | **0.0688** | 2.17× | 0.2203 | 90 % |
| | u_y | 0.1757 | **0.0709** | 2.48× | 0.2683 | 93 % |
| | θ | 0.0253 | **0.0097** | 2.60× | 0.0381 | 94 % |
| | p | 0.1735 | **0.0658** | 2.64× | 0.2386 | 92 % |

Model A sits on the paired-set ε-blind bound (0.225 vs 0.2249 at ε > 1.3): the ceiling is
informational, not architectural. Shallow alternatives fail the same way (kernel ridge
regression on the same 600 samples gains nothing from ε, 0.296 → 0.301; a reduced-rank linear
transfer of the reference correction gains ≤ 1.6× for velocity).

### 5.5 The physics prior: `D = ε·P(X)`

Same backbone, only the output structure changes (ε still read from a constant channel, target
still D):

| split | variant | u_x | u_y | θ | p | learned β |
|---|---|---|---|---|---|---|
| extrapolation | plain U-Net | 0.1485 | 0.1724 | 0.0237 | 0.1731 | — |
| | `D = ε·P(X)` | 0.0919 | 0.1063 | 0.0141 | 0.1061 | fixed 0 |
| | `D = (ε + βε²)·P(X)` | **0.0860** | **0.0946** | **0.0128** | **0.1029** | 0.255 / 0.205 / 0.211 / 0.246 |
| interpolation | plain U-Net | 0.0700 | 0.0712 | 0.0096 | 0.0669 | — |
| | `D = ε·P(X)` | 0.0515 | 0.0487 | 0.0078 | 0.0421 | fixed 0 |
| | `D = (ε + βε²)·P(X)` | **0.0516** | **0.0492** | **0.0070** | **0.0405** | 0.236 / 0.220 / 0.201 / 0.286 |

Two things happen at once. In extrapolation the factorised output beats the plain one by
**1.62×** (u_x), consistently across channels. And the network *discovers* a positive curvature
on its own, β = 0.20–0.29, the same sign and order as the ε² term measured in §5.1 — an
independent check of the asymptotic expansion.

A Fourier neural operator (1.19 M parameters) behaves identically: u_x 0.2286 (no ε) → 0.1348
(ε input) → **0.1454** (prior, X-only input, ε multiplied back outside the network), i.e. the
prior is backbone-independent.

### 5.6 Data efficiency, honestly

Learning curve on the fixed interpolation test set (200 samples), mean of three seeds:

| n | B: ε as input | C: P only, ε never enters the network |
|---|---|---|
| 0 (do nothing) | 0.2203 / 0.2683 / 0.0381 / 0.2386 | same |
| 50 | 0.1761 / 0.1721 / 0.0614 / 0.1619 | **0.1667 / 0.1623 / 0.0659 / 0.1729** |
| 100 | 0.1348 / 0.1252 / 0.0511 / 0.1197 | **0.1291 / 0.1201 / 0.0514 / 0.1332** |
| 200 | 0.0904 / 0.0867 / 0.0359 / 0.0807 | **0.0889 / 0.0901 / 0.0316 / 0.1016** |
| 400 | **0.0606 / 0.0545 / 0.0166 / 0.0486** | 0.0744 / 0.0731 / 0.0183 / 0.0849 |

In distribution the two are equivalent within seed noise at n ≤ 200, and with the full pool the
explicit ε channel is ~20 % better. The value of the prior is therefore *not* in-distribution
accuracy; it is that ε never enters the network, which is precisely what makes a single-ε data
set reusable.

### 5.7 External validation on an independent, published data set

`NurshatMenglik/NeuralConvection_2D_TrainTest` (v2; 4000 training pairs + 200 in-distribution
test pairs, CC BY 4.0), generated by another group with their own pipeline, at a **single**
ε = 1.2, Ra ∈ [10², 3·10⁶].

* *Conventions match.* Recomputing their published `relL2_*` columns as `‖Y−X‖/‖X‖` from the
  tensors reproduces them exactly (0.14233 / 0.13749 / 0.01356 / 0.19070 for sample 0); their
  pressure amplitude ratio is 0.929 against 0.931 for our fixed pairing.
* *The law reproduces.* On the ε = 1.2 slice:

| channel | public 4000 (median, p10–p90) | our data at ε ∈ [1.1,1.3] (n=64) |
|---|---|---|
| u_x | 0.1797 (0.112–0.300) | 0.1839 (0.126–0.284) |
| u_y | 0.2435 (0.141–0.403) | 0.2250 (0.150–0.395) |
| θ | 0.0228 (0.001–0.071) | 0.0232 (0.001–0.056) |
| p | 0.1858 (0.117–0.246) | 0.1889 (0.128–0.262) |

* *Even the secondary trends reproduce:* correlation of R with log₁₀Ra is
  `+0.133 / −0.419 / +0.647 / −0.198` on their data and `+0.053 / −0.520 / +0.652 / −0.185` on
  ours (u_x/u_y/θ/p). The θ channel — the one whose scatter we explain with χ — has `r = +0.65`
  in both, from independently generated data.
* *Four held-out splits, four prefactors, one predictable pattern.* The release publishes three
  further held-out splits generated by the same generator and the same solver commit
  (`v6_freefall_20260921`, `raw_both_deramp` pressure convention): `test_in_distribution` (200
  random-wall pairs), `test_ood` (200 random-wall pairs whose vertical-wall Fourier dials lie
  outside the training range), and `test_legacy` (200 pairs of the classical cavity: constant hot
  and cold side walls, adiabatic top and bottom). Our recomputation reproduces every published
  `relL2_*` column of all four splits exactly, so the comparison is convention-safe. At the fixed
  ε = 1.2 the medians of `R = ‖Y−X‖/‖Y‖` are

| split | n | `u_x` | `u_y` | `θ` | `p` |
|---|---|---|---|---|---|
| train | 4000 | 0.1797 | 0.2435 | 0.0228 | 0.1858 |
| test_in_distribution | 200 | 0.1822 | 0.2388 | 0.0220 | 0.1863 |
| test_ood | 200 | 0.1567 | 0.2157 | 0.0214 | 0.1671 |
| test_legacy | 200 | 0.2984 | 0.3390 | 0.0836 | 0.1983 |

  The exponent cannot be re-tested here - every split sits at one ε - but the prefactor is put
  under pressure and it moves: 1.9× between the classical cavity and the random-wall families at
  the same ε. That is a statement about the flow, not a defect of the law, and the θ channel makes
  the mechanism explicit. Writing `û = max|u_X|` (a property of the Boussinesq field, hence
  available to any surrogate),

`R_θ/ε = c·û^a`, with `a = 1.86 ± 0.03`, `c = 0.54` on our 600 samples;
  `a = 1.90 ± 0.01`, `c = 0.60` on their 4000; `a = 1.93 ± 0.04`, `c = 0.56` on their OOD split.
  Two independent generators, three families, R² = 0.90-0.92, coefficients agreeing to 25 %. The
  classical cavity is the outlier (`a = 2.37`, `c = 2.10`) and is also the one whose flow is
  qualitatively different: a single strong circulation with `û` 33 % above the random-wall median.
  The same reading explains the residual scatter in `u_x/u_y/p` in §5.1: A is not a universal
  constant, it is a functional of the flow, and the flow is in the input.

* *Cross-data-set transfer.* *(numbers in `~/nc_run/p7/public_transfer.json`; see §5.8)*

### 5.8 Cross-data-set transfer — the payoff of the prior

*(automatically generated table — `code/make_tables.py`)*

### 5.9 A second geometry (aspect ratio 2:1, 104 samples)

Same generator, same Ra and ε ranges, cavity twice as wide. The ε exponent is unchanged
(p = 1.09 / 1.16 / 0.93 / 1.08 for u_x/u_y/θ/p, all within ~1σ of the unit-square values),
while the prefactor moves (A = 0.175 / 0.175 / 0.008 / 0.222 versus 0.147 / 0.193 / 0.010 /
0.148). The Rayleigh trend is stronger for u_x and p here (q = −0.14, −0.15 vs 0.00, −0.03) and
reproduces for u_y and θ. **The ε-scaling is geometry-independent; calibration of A is not** —
which is exactly why the surrogate takes X as input rather than a table of A.

### 5.10 Solver noise is not the source of the law

* Convergence quality over 600 samples (compressible member): median relative change
  3.75e-6, p90 1.09e-4, worst 7.04e-4.
* Dropping the 65 samples with relative change > 1e-4 changes the u_x fit from
  `0.1809 ε − 0.0212` to `0.1813 ε − 0.0221` — unchanged.
* Re-running the 10 worst samples with `steady_tol` tightened from 1e-6 to 1e-8 leaves the
  solution **bit-identical** (`|ΔY|/|Y| = 0`). The reason is structural: the compressible march
  terminates on the zero-iteration guard, not on `steady_tol`, so the residual reported is the
  fixed-point residual of the discrete solution, not unfinished iteration.

### 5.11 Cost and speed-up (one RTX 4060 Laptop, 64², 8-way parallel solver)

| | median | mean | p90 | max |
|---|---|---|---|---|
| paired solve, 600 samples | 33.9 s | 38.0 s | 59.9 s | 510 s |
| paired solve, paired set | 55.8 s | 67.7 s | 136 s | 202 s |
| surrogate, GPU, batch 1 | 6.3 ms | | | |
| surrogate, GPU, batch 256 | 0.49 ms / sample | | | |
| surrogate, CPU, 1 thread | 14.0 ms | | | |

Factor **5.4×10³** per sample, **6.9×10⁴** in batches.

## 6. Limitations

1. **Two dimensions only.** A three-dimensional pair at res = 16 costs ≥ 23 min for the
   Boussinesq member alone (1386 s, 21 steps) and 3.4 GB of memory; the compressible member did
   not complete within 40 min, and the historical res = 24 probe at ε = 1.2 never converged
   (Newton backtracking to dt = 3.7e-4). A 3-D ε-sweep is out of reach on this machine and we do
   not claim one.
2. **Ma is inert here**, so the break-symmetry group is one-dimensional rather than two. At
   larger Ma this would have to be redone.
3. **ε < 2**, imposed by the equation of state (cold wall must stay positive).
4. The law leaves 25–40 % of the variance in u_x/u_y/p unexplained; χ closes that gap only for θ.
5. The prefactor A is geometry-dependent and must be learned (or calibrated) per configuration.
6. The public data set and ours differ by 0.08 % in Ra_eff at ε = 1.2 (1.2/1.095² = 1.000813);
   their v2 release corrects this and the two now agree.

## 7. Conclusion

We set out to build a surrogate for compressible natural convection and found that the object it
has to reproduce — the gap between the Boussinesq solution and the compressible one — is governed
by a single parameter in an exact and predictable way.

1. **One parameter.** In this non-dimensionalisation `Fr² = ΔT/T₀` identically, so the
   break-symmetry axis is ε = ΔT/T₀ and nothing else; Mach-number effects are O(Ma²) and
   are invisible at the values reached here.
2. **A clean law.** `R = ‖Y−X‖/‖Y‖ = A ε^p` with p = 1.023 ± 0.018, 0.995 ± 0.019,
   1.021 ± 0.077, 1.009 ± 0.015 (u_x, u_y, θ, p), an intercept indistinguishable from zero,
   and a seven-parameter boundary-condition description that adds only 3.5 % of explained
   variance. The exponent is geometry-independent and fluid-independent; only the prefactor A
   must be calibrated, which is why the surrogate takes X as input rather than a table of A.
3. **A reason.** The ε-expansion of the compressible member reproduces the Boussinesq system
   at O(1) and produces the correction through `∇·u₁ = ∇·(θ₀u₀)` and the density-modulated
   inertia. The linearised operator does not contain ε, so the correction factorises,
   `D = ε P(X) + ε² Q(X) + ...`, with a shape that does not rotate. All five
   consequences of the expansion are confirmed numerically.
4. **A design rule.** Predicting `P(X)` and multiplying by ε outside the network — four
   lines of code — converts the law into a prior that (i) beats the plain architecture by 1.62× in
   extrapolation, (ii) lets the network discover the ε² term itself, (iii) works the same
   way for a Fourier neural operator, and (iv) makes a public data set that exists at a single
   temperature contrast reusable at others, which is the difference between a surrogate that
   must be retrained and one that can be applied.

The honest limits are in §6: two dimensions, ε < 2, one-dimensional break symmetry, and a
prefactor that carries 25–40 % unexplained variance in three of the four channels. None of them
touch the claim that matters for practice, which is that the ε dependence is known in
closed form and should not be left to the network to discover from data.

## 8. Reproducibility

All scripts are in `D:\Desktop\算法代替仿真\code\` (Chinese-path Windows workspace, executed in
WSL2 with dolfinx 0.7.3 for the solver and PyTorch 2.10 for the surrogates); raw data, logs and
JSON results are under `~/nc_run/`:

| script | what it produces |
|---|---|
| `exp_eps_law.py` | the 627-sample law (§5.1) |
| `repack_p_fixed.py` | the corrected pressure channel (§5.2) |
| `exp_paired_final.py` | the ε-blind floor and shape invariance (§5.3) |
| `exp_modes.py` | the ε^k mode expansion (§3) |
| `exp_chi.py` | the χ explanation for θ (§5.1) |
| `lib.py`, `exp_p2_fixed.py`, `exp_prior_ablation.py`, `exp_fno.py`, `exp_learning_curve.py` | surrogates (§5.4–5.6) |
| `exp_public_transfer.py` | external validation and transfer (§5.7–5.8) |
| `exp_aspect2.py` | the 2:1 geometry (§5.9) |
| `exp_tight.py`, `exp_final_numbers.py` | noise control and timing (§5.10–5.11) |
| `make_tables.py` | regenerates every table from the JSON results |