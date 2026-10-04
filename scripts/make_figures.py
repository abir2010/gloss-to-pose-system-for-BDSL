"""Regenerate every figure from the CSVs in results/.   python scripts/make_figures.py
Needs numpy and matplotlib only."""
import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#eb6834", "#2f9e6f", "#c0392b", "#8a8a85"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "figure.facecolor": "white"})


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------ 1. main comparison
rows = read(RES / "main_comparison" / "architecture_comparison.csv")
names = ["Progressive\nTransformer", "NAT + MSE\n(final system)", "Diffusion\n(NAT+Diffusion\n+Velocity)"]
colors = [ORANGE, GREEN, RED]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
for a, (key, title, fmt) in zip(ax, [("dtw_mje", "DTW-MJE (lower is better)", "{:.3f}"),
                                      ("pck_0.5", "PCK@0.5, % (higher is better)", "{:.1f}"),
                                      ("motion_ratio", "Motion ratio (1.0 = reference)", "{:.2f}")]):
    vals = [float(r[key]) for r in rows]
    bars = a.bar(names, vals, color=colors, zorder=3)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    a.set_title(title, fontsize=9)
    a.grid(axis="y", color="#e6e6e2", zorder=0)
    a.tick_params(axis="x", labelsize=7.5)
plt.tight_layout()
plt.savefig(FIG / "main_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------ 2. training curves
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for a, (fname, title) in zip(ax, [("progressive", "Progressive Transformer"), ("nat_mse", "NAT + MSE"),
                                  ("diffusion", "Diffusion (NAT+Diffusion+Velocity)")]):
    h = read(RES / "training_histories" / f"{fname}.csv")
    ep = [int(r["epoch"]) for r in h]
    tr, va = [float(r["train"]) for r in h], [float(r["val"]) for r in h]
    a.plot(ep, tr, color=BLUE, label="train", lw=1.4)
    a.plot(ep, va, color=ORANGE, label="validation", lw=1.4)
    a.axvline(ep[int(np.argmin(va))], color=GREY, ls="--", lw=0.9)
    a.set_yscale("log")
    a.set_xlabel("epoch")
    a.set_title(f"{title}\n({len(ep)} epochs)", fontsize=9)
    a.grid(color="#e6e6e2")
ax[0].set_ylabel("loss (log scale)")
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("Loss scales differ across objectives; compare the shapes, not the magnitudes", fontsize=9, y=1.03)
plt.tight_layout()
plt.savefig(FIG / "training_curves.png", dpi=200, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------ 3. back-translation
bt = read(RES / "back_translation" / "summary.csv")
order = ["Real test poses", "Real test poses, smoothed (7-frame)", "NAT + MSE", "Progressive Transformer",
         "Diffusion (CFG 4)", "Diffusion (CFG 2.5)", "Diffusion (CFG 1)",
         "Wrong-sentence control (floor)"]
label = {"Diffusion (CFG 4)": "Diffusion, guidance 4.0", "Diffusion (CFG 2.5)": "Diffusion, guidance 2.5",
         "Diffusion (CFG 1)": "Diffusion, guidance 1.0", "Wrong-sentence control (floor)": "Wrong-sentence floor"}
by = {r["condition"]: r for r in bt}
fig, a = plt.subplots(figsize=(7.2, 3.6))
ys = np.arange(len(order))[::-1]
for y, c in zip(ys, order):
    r = by[c]
    w, lo, hi = float(r["wer_pct"]), float(r["wer_ci_lo"]), float(r["wer_ci_hi"])
    col = GREEN if c.startswith(("Real", "NAT")) else (ORANGE if c.startswith("Progressive") else RED)
    if c.startswith("Wrong"):
        col = GREY
    a.barh(y, w, xerr=[[w - lo], [hi - w]], color=col, zorder=3, error_kw={"lw": 1})
    a.text(hi + 2, y, f"{w:.1f}", va="center", fontsize=8)
a.set_yticks(ys)
a.set_yticklabels([label.get(c, c) for c in order], fontsize=8)
a.set_xlabel("word error rate of the recogniser on the generated poses, % (lower is better; 95% CI)")
a.set_xlim(0, 112)
a.grid(axis="x", color="#e6e6e2", zorder=0)
a.set_title("Back-translation (exploratory: the recogniser did not reach its 90% sentence-accuracy gate)", fontsize=8.5)
plt.tight_layout()
plt.savefig(FIG / "back_translation.png", dpi=200, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------ 4. velocity x guidance
grid = []
for f in sorted(glob.glob(str(RES / "ablation" / "cfg_sweep_vel*.csv"))):
    grid += read(f)
grid.sort(key=lambda r: (float(r["velocity_weight"]), float(r["cfg_scale"])))
with open(RES / "ablation" / "velocity_guidance_grid.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
    w.writeheader()
    w.writerows(grid)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
for c, vel in zip([RED, ORANGE, GREEN], sorted({float(r["velocity_weight"]) for r in grid})):
    g = [r for r in grid if float(r["velocity_weight"]) == vel]
    for a, k in zip(ax, ["motion_ratio", "jitter_ratio", "dtw_mje"]):
        a.plot([float(r["cfg_scale"]) for r in g], [float(r[k]) for r in g], marker="o", color=c,
               label=f"velocity weight {vel:g}")
for a, t in zip(ax, ["Motion ratio (1.0 = reference)", "Jitter ratio (1.0 = reference)", "DTW-MJE (lower is better)"]):
    a.set_title(t, fontsize=9)
    a.set_xlabel("classifier-free guidance scale")
    a.grid(color="#e6e6e2")
ax[0].axhline(1.0, color=GREY, ls="--", lw=0.8)
ax[1].axhline(1.0, color=GREY, ls="--", lw=0.8)
ax[0].legend(frameon=False, fontsize=7.5)
plt.tight_layout()
plt.savefig(FIG / "velocity_guidance.png", dpi=200, bbox_inches="tight")
plt.close()

print("wrote:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))
