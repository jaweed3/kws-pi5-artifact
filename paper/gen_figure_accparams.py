#!/usr/bin/env python3
"""Generate accuracy-vs-params scatter for the KWS paper (4 models, fp32)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# (label, params, fp32 accuracy)
MODELS = [
    ("DS-CNN", 22604, 90.55),
    ("DNN", 88396, 79.63),
    ("LSTM", 19980, 90.48),
    ("CRNN", 56844, 92.56),
]

names = [m[0] for m in MODELS]
params = np.array([m[1] for m in MODELS])
acc = np.array([m[2] for m in MODELS])
colors = ["#4C9BE8", "#E89B4C", "#50B85E", "#C44E52"]

fig, ax = plt.subplots(figsize=(6.6, 2.6))
for i, (n, p, a) in enumerate(MODELS):
    ax.scatter(p, a, s=70, color=colors[i], edgecolor="black", linewidth=0.5, zorder=3)
    ax.annotate(n, (p, a), textcoords="offset points", xytext=(8, 6),
                fontsize=7, va="bottom")
ax.set_xscale("log")
ax.set_xticks([2e4, 3e4, 4e4, 5e4, 6e4, 8e4])
ax.set_xticklabels(["20k", "30k", "40k", "50k", "60k", "80k"])
ax.set_xlabel("Parameters")
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_ylim(77, 95)
ax.set_title("Accuracy versus model size (float32, Raspberry Pi 5)")
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
out = os.path.join(FIGDIR, "acc_params.pdf")
plt.savefig(out)
print(f"figure: {out}")
