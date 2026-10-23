"""Two panels: (a) integral-Nu change versus eps for five Ra, isothermal walls;
(b) surrogate integral-Nu error versus eps for floor / prior / prior+flux."""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/nc_run/paired/dataset")
DS = os.path.expanduser("~/nc_run/paired/dataset")
OUT = os.path.join(os.path.dirname(HERE), "论文", "figs", "fig_nu_eps_ra.png")
M = 6


def scatter(vals, cfile, n):
    c = np.load(cfile)
    ux = np.unique(np.round(c[:, 0], 12))
    uy = np.unique(np.round(c[:, 1], 12))
    col = np.searchsorted(ux, np.round(c[:, 0], 12))
    row = np.searchsorted(uy, np.round(c[:, 1], 12))
    img = np.full((n + 1, n + 1), np.nan)
    img[row, col] = vals
    return img


def nu(d, kind, eps, n=64):
    T = np.load(os.path.join(d, "T_%s.npy" % kind)).astype(np.float64)
    th = scatter((T - (1 - eps / 2)) / eps, os.path.join(d, "coords_T_%s.npy" % kind), n)
    xw = np.arange(M) * (1.0 / (n - 1))
    w = np.linalg.solve(np.vander(xw, M, increasing=True).T, np.eye(M)[1])
    return float(-np.nanmean(th[:, :M].dot(w)))


tags = {}
for d in sorted(glob.glob(os.path.join(DS, "c*_Ra*_1100_e*"))):
    m = re.match(r"(c\d+_Ra\S+?)_e([\d.]+)$", os.path.basename(d))
    if m and os.path.isdir(os.path.join(d, "sample_000000")):
        tags.setdefault(m.group(1), {})[float(m.group(2))] = os.path.join(
            d, "sample_000000")

curves = {}
for k, v in tags.items():
    ra = int(float(re.search(r"Ra(\S+?)_", k).group(1)))
    xs, ys = [], []
    for e, d in sorted(v.items()):
        y, x = nu(d, "comp", e), nu(d, "bouss", e)
        xs.append(e)
        ys.append(100 * (y - x) / x)
    curves[ra] = (np.array(xs), np.array(ys))

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=240)
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(curves)))
for c, (ra, (x, y)) in zip(colors, sorted(curves.items())):
    ax[0].plot(x, y, "o-", ms=3.2, lw=1.2, color=c, label=r"$Ra=10^{%d}$" % int(np.log10(ra)))
ax[0].axhline(0, color="0.4", lw=.8)
ax[0].set_xlabel(r"$\varepsilon$")
ax[0].set_ylabel(r"$100\,(Nu_Y-Nu_X)/Nu_X$")
ax[0].set_title("(a) integral change, isothermal walls", fontsize=9)
ax[0].grid(alpha=.25)
ax[0].legend(fontsize=7, ncol=2)

j = json.load(open(os.path.join(os.path.dirname(HERE), "论文", "flux_cross_eps.json")))
bins = sorted(j["bins"], key=float)
for tag, col, mk in [("do-nothing", "#7f8c8d", "o"),
                     ("prior", "#c0392b", "s"),
                     ("prior+flux", "#1f77b4", "^")]:
    x = np.array([float(b) for b in bins])
    y = np.array([100 * j["bins"][b][tag]["left"] for b in bins])
    se = np.array([100 * j["bins"][b][tag]["left_sem"] for b in bins])
    ax[1].errorbar(x, y, yerr=se, marker=mk, color=col, lw=1.3, ms=5,
                   capsize=2.5, label={"do-nothing": "return $X$",
                                        "prior": "field-only prior",
                                        "prior+flux": "prior + wall-flux"}[tag])
ax[1].set_xlabel(r"$\varepsilon$")
ax[1].set_ylabel("hot-wall $Nu$ error (%)")
ax[1].set_title("(b) surrogate error, 600 random walls", fontsize=9)
ax[1].grid(alpha=.25)
ax[1].legend(fontsize=7)
for a in ax:
    a.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
