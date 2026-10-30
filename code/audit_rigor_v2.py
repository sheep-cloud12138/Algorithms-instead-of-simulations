"""Audits added after review.

1. Independent wall-flux evaluation of a stencil-free scaling model fitted only
   on paired fields from data_public/pairs600_fixedp.npz.
2. Seed-level confidence intervals for the public and within-family wall-flux
   experiments already stored in the repository.
3. Direct 4/6/8/10-point evaluation of the recovered wall-flux U-Nets.
4. A compact summary of the existing 3D leave-one-epsilon ROM check.
5. Grid-resolution uncertainty inferred from the existing Nu grid runs.

The script uses no simulator. It reads the NumPy re-evaluation of the recovered
public checkpoints for the direct stencil audit. It writes
论文/rigor_audit.json and 论文/figs/fig_stencil_audit.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_public" / "pairs600_fixedp.npz"
PAPER = ROOT / "论文"
OUT = PAPER / "rigor_audit.json"
FIG = PAPER / "figs" / "fig_stencil_audit.png"
STENCILS = (4, 6, 8, 10)


def split_indices(eps: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "extrap":
        return eps <= 1.3, eps > 1.3
    order = np.argsort(eps)
    test = np.zeros(len(eps), dtype=bool)
    test[order[::3]] = True
    return ~test, test


def stencil_weights(n: int, points: int) -> np.ndarray:
    x = np.arange(points) / (n - 1)
    return np.linalg.solve(
        np.vander(x, points, increasing=True).T,
        np.eye(points)[1],
    )


def wall_flux(theta: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = len(weights)
    hot = -(theta[:, :, :m] * weights).sum(-1)
    cold = (theta[:, :, -m:][..., ::-1] * weights).sum(-1)
    return hot, cold


def q_errors(
    pred_theta: np.ndarray,
    true_theta: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hot_p, cold_p = wall_flux(pred_theta, weights)
    hot_t, cold_t = wall_flux(true_theta, weights)
    hot = np.abs(hot_p - hot_t).mean(1) / np.abs(hot_t).mean(1)
    cold = np.abs(cold_p - cold_t).mean(1) / np.abs(cold_t).mean(1)
    both = (
        np.abs(hot_p.mean(1) - hot_t.mean(1))
        + np.abs(cold_p.mean(1) - cold_t.mean(1))
    ) / (np.abs(hot_t.mean(1)) + np.abs(cold_t.mean(1)))
    return hot, cold, both


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(values), (4000, len(values)))
    means = values[idx].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def fit_scaling(
    x: np.ndarray,
    y: np.ndarray,
    eps: np.ndarray,
    train: np.ndarray,
) -> np.ndarray:
    target = (y[train] - x[train]) / eps[train, None, None, None]
    source = x[train]
    return (target * source).sum((0, 2, 3)) / (source**2).sum((0, 2, 3))


def stencil_audit() -> dict:
    z = np.load(DATA, allow_pickle=True)
    x = z["X"].astype(np.float64)
    y = z["Y"].astype(np.float64)
    meta = json.loads(str(z["meta"]))
    eps = np.array([item["eps"] for item in meta], dtype=np.float64)
    theta_x = (x[:, 2] - (1.0 - eps / 2.0)[:, None, None]) / eps[:, None, None]
    theta_y = (y[:, 2] - (1.0 - eps / 2.0)[:, None, None]) / eps[:, None, None]
    rng = np.random.default_rng(20261030)
    result = {}

    for split in ("extrap", "interp"):
        train, test = split_indices(eps, split)
        alpha = fit_scaling(x, y, eps, train)
        pred = x[test] + eps[test, None, None, None] * alpha[None, :, None, None] * x[test]
        pred_theta = (
            pred[:, 2] - (1.0 - eps[test] / 2.0)[:, None, None]
        ) / eps[test, None, None]
        field_floor = np.linalg.norm(theta_x[test] - theta_y[test], axis=(1, 2)) / np.linalg.norm(
            theta_y[test], axis=(1, 2)
        )
        field_model = np.linalg.norm(pred_theta - theta_y[test], axis=(1, 2)) / np.linalg.norm(
            theta_y[test], axis=(1, 2)
        )
        split_result = {
            "n_test": int(test.sum()),
            "coefficients": alpha.tolist(),
            "field": {
                "floor": {
                    "median": float(np.median(field_floor)),
                    "mean": float(field_floor.mean()),
                },
                "scaling_only": {
                    "median": float(np.median(field_model)),
                    "mean": float(field_model.mean()),
                    "mean_ci95": bootstrap_ci(field_model, rng),
                },
            },
            "stencils": {},
        }
        for points in STENCILS:
            weights = stencil_weights(theta_x.shape[-1], points)
            floor_hot, floor_cold, floor_both = q_errors(theta_x[test], theta_y[test], weights)
            model_hot, model_cold, model_both = q_errors(pred_theta, theta_y[test], weights)
            split_result["stencils"][str(points)] = {
                "floor": {
                    "hot_mean": float(floor_hot.mean()),
                    "cold_mean": float(floor_cold.mean()),
                    "both_mean": float(floor_both.mean()),
                    "both_median": float(np.median(floor_both)),
                },
                "scaling_only": {
                    "hot_mean": float(model_hot.mean()),
                    "cold_mean": float(model_cold.mean()),
                    "both_mean": float(model_both.mean()),
                    "both_median": float(np.median(model_both)),
                    "both_mean_ci95": bootstrap_ci(model_both, rng),
                    "fraction_field_lt_1_nu_gt_2": float(
                        ((field_model < 0.01) & (model_both > 0.02)).mean()
                    ),
                },
            }
        result[split] = split_result
    return result


def unet_stencil_audit() -> dict:
    previous = json.loads((PAPER / "public_flux_stencil.json").read_text())
    current = json.loads((PAPER / "public_flux_stencil_10pt.json").read_text())
    reproduction_differences = []
    for model in previous:
        reproduction_differences.append(
            abs(previous[model]["relL2_T"] - current[model]["relL2_T"])
        )
        for points in ("M4", "M6", "M8"):
            for wall in ("hot", "cold", "floor"):
                reproduction_differences.append(
                    abs(previous[model][points][wall] - current[model][points][wall])
                )
    result = {
        "sample_count": 200,
        "split": "public legacy classical cavity, held out",
        "wall_flux_seeds": [0, 1],
        "reproduction_max_abs_difference": float(max(reproduction_differences)),
        "models": {},
    }
    for model, row in current.items():
        result["models"][model] = {
            "relL2_T": row["relL2_T"],
            "stencils": {
                points: {
                    "hot": row[points]["hot"],
                    "cold": row[points]["cold"],
                    "both_mean": 0.5 * (row[points]["hot"] + row[points]["cold"]),
                    "floor_hot": row[points]["floor"],
                }
                for points in ("M4", "M6", "M8", "M10")
            },
        }
    result["interpretation"] = (
        "The matched six-point gain does not persist consistently across both walls "
        "and both trained seeds at ten points. Seed 0 is worse than returning X in "
        "the both-wall mean, while seed 1 is only slightly better."
    )
    return result


def seed_statistics() -> dict:
    public = json.loads((PAPER / "seed_sweep.json").read_text())
    seeds = sorted(public["seeds"], key=int)
    result = {"public_legacy": {}, "within_family": {}}
    rng = np.random.default_rng(20261030)
    for kind in ("pub_plain", "pub_prior", "pub_flux"):
        entry = {}
        for metric, key in (("temperature_field", "_T"), ("both_wall_Nu", "_Nu"), ("local_flux", "_local")):
            arr = np.array([public["seeds"][s][kind]["legacy"][key] for s in seeds])
            seed_means = arr.mean(1)
            entry[metric] = {
                "seed_means": seed_means.tolist(),
                "mean": float(seed_means.mean()),
                "sd": float(seed_means.std(ddof=1)),
                "ci95_t": [
                    float(seed_means.mean() - 2.776 * seed_means.std(ddof=1) / np.sqrt(len(seed_means))),
                    float(seed_means.mean() + 2.776 * seed_means.std(ddof=1) / np.sqrt(len(seed_means))),
                ],
            }
        result["public_legacy"][kind] = entry
    prior = np.array([public["seeds"][s]["pub_prior"]["legacy"]["_Nu"] for s in seeds]).mean(0)
    flux = np.array([public["seeds"][s]["pub_flux"]["legacy"]["_Nu"] for s in seeds]).mean(0)
    difference = prior - flux
    result["public_legacy"]["paired_prior_minus_flux_Nu"] = {
        "mean": float(difference.mean()),
        "median": float(np.median(difference)),
        "mean_ci95_bootstrap": bootstrap_ci(difference, rng),
        "positive_fraction": float((difference > 0).mean()),
    }

    own = json.loads((PAPER / "own_seed_sweep.json").read_text())
    own_seeds = sorted(own["seeds"], key=int)
    for kind in ("own_prior", "own_flux"):
        entry = {}
        for metric in ("T", "Nu_both", "local"):
            values = np.array([own["seeds"][s][kind]["eps12"][metric]["median"] for s in own_seeds])
            entry[metric] = {
                "seed_values": values.tolist(),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "ci95_t": [
                    float(values.mean() - 2.776 * values.std(ddof=1) / np.sqrt(len(values))),
                    float(values.mean() + 2.776 * values.std(ddof=1) / np.sqrt(len(values))),
                ],
            }
        result["within_family"][kind] = entry
    return result


def grid_uncertainty() -> dict:
    eps12 = json.loads((PAPER / "nu_grid_ra1e6_256.json").read_text())
    eps19 = json.loads((PAPER / "nu_grid_ra1e6_192.json").read_text())
    return {
        "Ra1e6_eps1.2": {
            "dNu_by_grid": eps12["rel"],
            "richardson_limit": eps12["richardson_limit"],
            "coarse_target_bias": float(eps12["rel"]["64"] - eps12["richardson_limit"]),
            "finest_pair_change": float(eps12["rel"]["256"] - eps12["rel"]["128"]),
        },
        "Ra1e6_eps1.9": {
            "dNu_by_grid": eps19["rel"],
            "finest_pair_change": float(eps19["rel"]["192"] - eps19["rel"]["128"]),
            "coarse_target_bias_vs_192": float(eps19["rel"]["64"] - eps19["rel"]["192"]),
        },
    }


def three_dimensional_audit() -> dict:
    data = json.loads((PAPER / "3d_rom_eps.json").read_text())
    agg = data["aggregate"]
    out = {
        "n_train": int(data["rows"][0]["n_train"]),
        "n_held_out_per_epsilon": int(data["rows"][0]["n_test"]),
        "epsilon_count": len(data["rows"]),
        "channels": {},
    }
    for idx, channel in enumerate(("ux", "uy", "theta", "p")):
        floor = agg["floor"]["mean"][idx]
        prior = agg["prior_m3"]["mean"][idx]
        out["channels"][channel] = {
            "floor_mean": float(floor),
            "prior_m3_mean": float(prior),
            "gain_vs_floor": float(floor / prior),
        }
    out["interpretation"] = (
        "Field-only 3D leave-one-epsilon ROM; no 3D wall-flux target is available."
    )
    return out


def plot_stencils(result: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), sharey=True)
    for ax, split in zip(axes, ("extrap", "interp")):
        points = [int(key) for key in result[split]["stencils"]]
        floor = [result[split]["stencils"][str(p)]["floor"]["both_mean"] for p in points]
        model = [result[split]["stencils"][str(p)]["scaling_only"]["both_mean"] for p in points]
        ax.plot(points, floor, "o-", label="return $X$", color="0.45")
        ax.plot(points, model, "s-", label="scaling-only field map", color="#1f77b4")
        ax.set_xticks(points)
        ax.set_xlabel("wall-gradient stencil points")
        ax.set_title("extrapolation" if split == "extrap" else "interpolation")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("mean both-wall $Nu$ error")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Scaling-only model evaluated without using a wall stencil in training", fontsize=10)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=300)
    plt.close(fig)


def main() -> None:
    result = {
        "stencil_independent": stencil_audit(),
        "unet_stencil": unet_stencil_audit(),
        "seed_statistics": seed_statistics(),
        "grid_uncertainty": grid_uncertainty(),
        "three_dimensional_holdout": three_dimensional_audit(),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_stencils(result["stencil_independent"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved", OUT)
    print("saved", FIG)


if __name__ == "__main__":
    main()
