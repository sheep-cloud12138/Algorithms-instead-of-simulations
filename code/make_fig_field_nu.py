"""Scatter of per-sample temperature-field error against integral-Nu error."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(HERE), "后续方案", "prelim_field_vs_nu.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(HERE), "论文", "figs", "fig_field_vs_nu.png")

j = json.load(open(SRC))
rows = j["rows"]
series = [
    ("return $X$", "floor", "#7f8c8d", "o"),
    ("field-only prior", "prior", "#c0392b", "s"),
    ("prior + wall-flux loss", "prior+flux", "#1f77b4", "^"),
]

fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=220)
for label, tag, color, marker in series:
    t = np.array([r[tag + "_T"] for r in rows])
    n = np.array([r[tag + "_Nu"] for r in rows])
    ax.scatter(t, n, s=24, alpha=.55, color=color, marker=marker,
               linewidths=0, label=label)

ax.axvline(0.01, color="0.35", lw=1.0, ls="--")
ax.axhline(0.02, color="0.35", lw=1.0, ls="--")
ax.fill_between([2e-4, 0.01], 0.02, 0.2, color="#f1c40f", alpha=.13, zorder=0)
ax.text(4e-4, 0.07, "field looks right,\nheat is wrong", fontsize=9, color="#7d6608")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(2e-4, 0.2)
ax.set_ylim(2e-4, 0.2)
ax.set_xlabel(r"temperature-field error  $\|Y-X\|_2/\|Y\|_2$", fontsize=11)
ax.set_ylabel(r"integral-$Nu$ error", fontsize=11)
ax.grid(alpha=.25, which="both")
ax.legend(fontsize=9, loc="lower right", framealpha=.95)
ax.set_title("200 held-out classical-cavity samples, $\\varepsilon=1.2$", fontsize=11)
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
