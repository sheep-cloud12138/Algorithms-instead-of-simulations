"""Summarise the complete training-order x evaluation-stencil matrix."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "论文" / "stencil_ckpt_sweep.json"
OUT = ROOT / "论文" / "stencil_matrix_summary.json"

MODEL_ORDER = {
    "pub_flux_lam1.0_s0_m4.pt": 4,
    "pub_flux_lam1.0_s1_m4.pt": 4,
    "pub_flux_lam1.0.pt": 6,
    "pub_flux_lam1.0_s1.pt": 6,
    "pub_flux_lam1.0_s0_m8.pt": 8,
    "pub_flux_lam1.0_s1_m8.pt": 8,
    "pub_flux_lam1.0_s0_m10.pt": 10,
    "pub_flux_lam1.0_s1_m10.pt": 10,
}


def seed_of(name):
    match = re.search(r"_s(\d+)(?:_m\d+)?\.pt$", name)
    return int(match.group(1)) if match else 0


def main():
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    rows = {}
    for name, values in sweep.items():
        if name in MODEL_ORDER:
            order = MODEL_ORDER[name]
            rows.setdefault(order, {})[seed_of(name)] = {
                "checkpoint": name,
                "m4": values["4pt"],
                "m6": values["6pt"],
                "m8": values["8pt"],
                "m10": values["10pt"],
            }

    complete = {}
    for order, seeds in sorted(rows.items()):
        if sorted(seeds) != [0, 1]:
            raise RuntimeError(f"incomplete seeds for train order {order}: {sorted(seeds)}")
        summary = {}
        for col in (4, 6, 8, 10):
            values = np.array([seed[f"m{col}"] for seed in seeds.values()])
            summary[f"m{col}"] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        complete[str(order)] = {"seeds": seeds, "across_seed": summary}

    blend = {
        name: {
            "m4": value["4pt"],
            "m6": value["6pt"],
            "m8": value["8pt"],
            "m10": value["10pt"],
        }
        for name, value in sweep.items()
        if name.startswith("pub_flux_blend")
    }
    OUT.write_text(
        json.dumps({"complete_diagonal": complete, "blend_mean_target": blend}, indent=1),
        encoding="utf-8",
    )
    print(json.dumps({"complete_diagonal": complete, "blend_mean_target": blend}, indent=1))


if __name__ == "__main__":
    main()
