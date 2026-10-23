"""Regenerate Figure 3 with the fair plain-times-eps/eps0 baseline."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P7 = os.path.expanduser("~/nc_run/p7")
CH = [r"$u_x$", r"$u_y$", r"$\theta$", r"$p$"]


def load(name):
    with open(os.path.join(P7, name)) as f:
        return json.load(f)


corrected = json.load(open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "论文", "transfer_corrected.json")))
same = corrected["same"]
cross = corrected["cross"]

fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.8), dpi=170)
col = ["#8c8c8c", "#e67e22", "#27ae60", "#2c6fbb"]

for i, (data, title) in enumerate(
    [
        (same, "same boundary-condition family (public 4000, random walls)"),
        (cross, "different boundary-condition family (public 5000, classical cavity)"),
    ]
):
    for j, (split, split_label) in enumerate(
        [("extrap", r"our $\varepsilon>1.3$"), ("interp", r"our full $\varepsilon$ range")]
    ):
        a = ax[i][j]
        x = np.arange(4)
        if i == 0:
            series = [
                (data[split]["plain"], "plain", col[0]),
                (data[split]["plain_rescaled"], r"plain $\times\,\varepsilon/\varepsilon_0$", col[1]),
                (data[split]["prior"], r"prior $D=\varepsilon P(X)$", col[2]),
                (data[split]["own_prior"], "own data, prior", col[3]),
            ]
        else:
            series = [
                (data[split]["plain"], "plain", col[0]),
                (data[split]["prior"], r"prior $D=\varepsilon P(X)$", col[2]),
                (data[split]["own_prior"], "own data, prior", col[3]),
            ]
        width = 0.8 / len(series)
        for q, (g, label, color) in enumerate(series):
            a.bar(x + (q - (len(series) - 1) / 2) * width, g, width, label=label, color=color)
        a.plot(x, data[split]["floor"], "k_", ms=22, mew=2, label="do nothing")
        a.set_xticks(x)
        a.set_xticklabels(CH, fontsize=12)
        a.set_title(title + "\n" + split_label, fontsize=9.5)
        a.set_ylabel(r"relative $L_2$ error", fontsize=10)
        a.grid(alpha=0.2, axis="y")
        if i == 0 and j == 0:
            a.legend(fontsize=8, ncol=2)

fig.suptitle(
    r"Applying the $\varepsilon$ scaling makes a single-contrast data set reusable within its family",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "/mnt/d/Desktop/算法代替仿真/论文/figs/fig6_transfer.png"
fig.savefig(out)
print("saved", out)
