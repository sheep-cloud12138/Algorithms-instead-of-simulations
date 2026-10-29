# Data and code deposit

Data, results and scripts for **"Field accuracy is not heat accuracy: wall-flux-constrained
surrogates for steady two-dimensional non-Boussinesq convection"** (International Journal of
Heat and Mass Transfer submission).

## Contents

| Path | Content |
|---|---|
| `data/p2` | 600 main paired Boussinesq/low-Mach solutions, 134 wall-mode families, `Ra~logU[1e2,1e6]`, `eps~U[0.2,1.9]` (`pairs600_fixedp.npz`) |
| `data/paired_dataset` | 20 flows x 5 contrasts (`eps=0.4..1.9`) exact paired set |
| `data/eps0_dataset` | 27 low-contrast checks (`eps=0.02/0.05/0.10`) |
| `data/water_dataset` | 104 pairs at `Pr=4.38`, bit-paired with the air cases |
| `data/cavity2_dataset` | 104 pairs on the 2:1 aspect-ratio cavity |
| `data/d3*` | three-dimensional paired checks at `Ra=1e4`, `3e4`, `1e5` (33 pairs total) |
| `results/p7` | every result JSON behind the numbers in the paper and supplement (each figure/table cites its source file) |
| `code` | full repository snapshot: dataset generation shell scripts, solver post-processing, experiments, figure/table scripts, audits |
| `SHA256SUMS` | checksums of every file |

The public data set of Menglik et al. used for the external-replication and transfer
experiments is third-party (CC BY 4.0) and is re-downloadable with `code/research/dl.py`.

## Reproducing

Environment: WSL2 Ubuntu 22.04. Solver: dolfinx 0.7.3; networks: PyTorch 2.x + CUDA.
See `code/README.md` for the full pipeline. Entry points for the paper's headline results:

```bash
python exp_eps_law.py         # scaling law on 627 pairs
python exp_field_vs_nu.py     # per-sample field error vs Nu error
python exp_ml_final.py        # ablations (3 seeds)
python exp_fair_scaling.py    # fair plain x eps/eps0 baseline
python make_tables.py         # all tables from JSONs
python make_figs.py; python make_figs2.py; python fig1_law.py; python make_fig9.py   # all figures
python audit_numbers.py       # recompute numbers quoted in the text
```

## License

<!-- TODO(author): choose licenses before uploading: -->
- Code: MIT (suggested).
- Data: CC BY 4.0 (suggested).
