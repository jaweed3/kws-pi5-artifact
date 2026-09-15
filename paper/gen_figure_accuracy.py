#!/usr/bin/env python3
"""Generate accuracy figure (fp32 vs int8 per model) for the KWS paper."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "bench", "results")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# model -> (label, fp32 accuracy, int8 accuracy, params)
MODELS = [
    ("DS-CNN", 90.55, 90.45, 22604),
    ("DNN", 79.63, 79.65, 88396),
    ("LSTM", 90.48, 87.43, 19980),
    ("CRNN", 92.56, 92.17, 56844),
]


def main():
    names = [m[0] for m in MODELS]
    fp32 = [m[1] for m in MODELS]
    i8 = [m[2] for m in MODELS]
    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    b1 = ax.bar(x - width / 2, fp32, width, label="float32",
                color="#4C9BE8", edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + width / 2, i8, width, label="int8 (quantized)",
                color="#E89B4C", edgecolor="black", linewidth=0.4)

    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(75, 95)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Accuracy over Speech Commands v2 test set (10 commands, 4,074 clips)")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "accuracy.pdf")
    plt.savefig(out)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
