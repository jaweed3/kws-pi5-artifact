#!/usr/bin/env python3
"""Generate multi-model latency figure for the KWS paper.

Reads results JSONs from bench/results/ (actual structure: bench_{model}_{prec}.json
with {"tflite": {...}, "onnx": {...}, "params": N}) and produces:
  figures/latency_multi.pdf  (grouped bars: model x runtime)
"""
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

# model -> (label, fp32 json, int8 json, params)
MODELS = [
    ("DS-CNN", "benchmark_kws_pi.json", "benchmark_mixed_pi.json", 22604),
    ("DNN", "bench_dnn_fp32.json", "bench_dnn_int8.json", 88396),
    ("LSTM", "bench_lstm_fp32.json", "bench_lstm_int8.json", 19980),
    ("CRNN", "bench_crnn_fp32.json", "bench_crnn_int8.json", 56844),
]
RUNTIMES = ["TFLite fp32", "TFLite int8", "ONNX fp32"]
COLORS = {"TFLite fp32": "#4C9BE8", "TFLite int8": "#E89B4C", "ONNX fp32": "#50B85E"}


def load(fname):
    p = os.path.join(RESULTS, fname)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    data = {}
    for name, fp32f, int8f, params in MODELS:
        fp = load(fp32f)
        i8 = load(int8f)
        entry = {"mean": {}, "p99": {}, "fps": {}, "params": params}
        if fp is None:
            print(f"  (missing {fp32f})")
        else:
            entry["mean"]["TFLite fp32"] = fp["tflite"]["mean_ms"]
            entry["p99"]["TFLite fp32"] = fp["tflite"]["p99_ms"]
            entry["fps"]["TFLite fp32"] = fp["tflite"]["fps"]
            entry["mean"]["ONNX fp32"] = fp["onnx"]["mean_ms"]
            entry["p99"]["ONNX fp32"] = fp["onnx"]["p99_ms"]
            entry["fps"]["ONNX fp32"] = fp["onnx"]["fps"]
        if i8 is not None:
            entry["mean"]["TFLite int8"] = i8["tflite"]["mean_ms"]
            entry["p99"]["TFLite int8"] = i8["tflite"]["p99_ms"]
            entry["fps"]["TFLite int8"] = i8["tflite"]["fps"]
        data[name] = entry

    names = list(data.keys())
    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for i, rt in enumerate(RUNTIMES):
        means = [data[n]["mean"].get(rt, np.nan) for n in names]
        ax.bar(x + (i - 1) * width, means, width, label=rt,
               color=COLORS[rt], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Inference latency by model and runtime (Raspberry Pi 5)")
    ax.legend(fontsize=7, ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    for i, n in enumerate(names):
        for j, rt in enumerate(RUNTIMES):
            v = data[n]["mean"].get(rt, None)
            if v is not None:
                ax.annotate(f"{v:.3f}", (x[i] + (j - 1) * width, v),
                            ha="center", va="bottom", fontsize=6)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "latency_multi.pdf")
    plt.savefig(out)
    print(f"figure: {out}")

    for n in names:
        m = data[n]["mean"]
        ratio = m.get("ONNX fp32", None) and m.get("TFLite fp32", None)
        if ratio:
            print(f"  {n}: TFLite fp32 {m['TFLite fp32']:.4f} ms | "
                  f"int8 {m['TFLite int8']:.4f} ms | ONNX {m['ONNX fp32']:.4f} ms | "
                  f"ratio {m['ONNX fp32']/m['TFLite fp32']:.2f}")


if __name__ == "__main__":
    main()
