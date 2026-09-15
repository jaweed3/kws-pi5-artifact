#!/usr/bin/env python3
"""KWS inference benchmark on Raspberry Pi 5: TFLite (ai-edge-litert) vs ONNX Runtime.

Paper A: "Benchmarking Keyword Spotting Inference on ARM64 Edge Devices:
TensorFlow Lite vs ONNX Runtime on Raspberry Pi 5"

Protocol (inherited from ComTech / drone_face edge bench):
  - sequential execution, NO parallelism (parallel inflates 2-4x)
  - WARMUP + MEASURED iterations on a fixed input (shape [1,49,10,1] / [1,1,49,10])
  - raw latency samples stored -> mean/median/p95/p99/std
  - IQR + 1.5*IQR outlier removal, 10k bootstrap 95% CI, seed 42
  - resource: RSS (psutil), CPU%, SoC temp (vcgencmd), governor reported

Also runs a top-1 accuracy pass over the same test wavs through both runtimes
to prove both graphs compute the same function.

Usage:
  python benchmark_kws.py [--iters 1000] [--warmup 1000] [--output results/benchmark_kws.json]
"""
import argparse
import json
import platform
import statistics
import subprocess
import time

import numpy as np
import psutil

from mfcc import mfcc_from_wav

LABELS = ["Down", "Go", "Left", "No", "Off", "On", "Right",
          "Stop", "Up", "Yes", "Silence", "Unknown"]

DEFAULT_TFLITE = "models/kws_ref_model_float32.tflite"
DEFAULT_ONNX = "models/kws_ref_model.onnx"
DEFAULT_WAVLIST = "data/testlist.tsv"


def read_temp():
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True,
                             text=True, timeout=2).stdout
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None


def governor():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except Exception:
        return None


def set_governor(gov):
    try:
        for cpu in range(4):
            p = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            subprocess.run(["sudo", "tee", p], input=f"{gov}\n".encode(),
                           capture_output=True, check=True)
        return True
    except Exception:
        return False


def load_tflite(path):
    import ai_edge_litert.interpreter as tfl
    interp = tfl.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    def run(x):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(out["index"])
    return run, (interp, inp, out)


def load_onnx(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    def run(x):
        return sess.run(None, {iname: x})[0]
    return run, (sess, ishape)


def prepare_input(feat, backend, ishape=None):
    x = feat[None, ...].astype(np.float32)              # [1,49,10,1] NHWC
    if backend == "onnx" and ishape and ishape[1] == 1 and ishape[2] != 1:
        x = np.transpose(x, (0, 3, 1, 2))               # -> NCHW [1,1,49,10]
    return x


def stats(samples):
    samples = np.asarray(samples, dtype=np.float64)
    q1, q3 = np.percentile(samples, [25, 75])
    iqr = q3 - q1
    clean = samples[(samples >= q1 - 1.5 * iqr) & (samples <= q3 + 1.5 * iqr)]
    rng = np.random.default_rng(42)
    boots = [np.mean(rng.choice(clean, size=len(clean), replace=True))
             for _ in range(10000)]
    ci = np.percentile(boots, [2.5, 97.5])
    return {
        "n_raw": int(len(samples)),
        "n_clean": int(len(clean)),
        "n_outliers": int(len(samples) - len(clean)),
        "mean_ms": float(np.mean(clean) * 1000),
        "median_ms": float(np.median(clean) * 1000),
        "p95_ms": float(np.percentile(clean, 95) * 1000),
        "p99_ms": float(np.percentile(clean, 99) * 1000),
        "std_ms": float(np.std(clean) * 1000),
        "ci95_ms": [float(ci[0] * 1000), float(ci[1] * 1000)],
        "fps": float(1.0 / np.mean(clean)),
    }


def accuracy(run_fn, backend, wavlist, limit=0):
    rows = []
    with open(wavlist) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p, lab = line.split("\t")
            rows.append((p, int(lab)))
    if limit:
        rows = rows[:limit]
    correct = 0
    feat_ms = 0.0
    t0 = time.perf_counter()
    for p, lab in rows:
        f = mfcc_from_wav(p)
        logits = run_fn(prepare_input(f, backend))[0]
        if int(np.argmax(logits)) == lab:
            correct += 1
    wall = time.perf_counter() - t0
    return {"n": len(rows), "top1": correct / len(rows), "wall_s": wall}


def bench_one(run_fn, backend, x, iters, warmup):
    # warmup
    for _ in range(warmup):
        run_fn(x)
    # measured
    lat = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        run_fn(x)
        lat[i] = time.perf_counter() - t0
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default=DEFAULT_TFLITE)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--wavlist", default=DEFAULT_WAVLIST)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--acc-limit", type=int, default=0)
    ap.add_argument("--output", default="results/benchmark_kws.json")
    ap.add_argument("--tag", default=None, help="model tag written into JSON")
    ap.add_argument("--params", type=int, default=None,
                    help="parameter count written into JSON")
    ap.add_argument("--governor", default="performance",
                    help="set cpufreq governor for the run (restored after)")
    args = ap.parse_args()

    report = {
        "meta": {
            "device": platform.node(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "kernel": platform.release(),
            "governor_before": governor(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": "warmup+measured, sequential, seed=42, IQR outliers, 10k bootstrap CI",
        },
        "iters": args.iters,
        "warmup": args.warmup,
        "models": {"tflite": args.tflite, "onnx": args.onnx},
    }
    if args.tag:
        report["tag"] = args.tag
    if args.params:
        report["params"] = args.params

    gov_before = governor()
    set_gov = (args.governor and gov_before != args.governor
               and set_governor(args.governor))
    report["meta"]["governor_during"] = governor()

    # fixed input from first wav in the list
    feat = None
    with open(args.wavlist) as f:
        for line in f:
            line = line.strip()
            if line:
                feat = mfcc_from_wav(line.split("\t")[0])
                break

    print(f"[*] device={report['meta']['machine']} governor={report['meta']['governor_during']}")

    run_tfl, _ = load_tflite(args.tflite)
    run_onnx, (_, ishape_onnx) = load_onnx(args.onnx)
    x_tfl = prepare_input(feat, "tflite")
    x_onnx = prepare_input(feat, "onnx", ishape_onnx)

    # sanity: outputs agree
    o1 = run_tfl(x_tfl)[0]
    o2 = run_onnx(x_onnx)[0]
    maxdiff = float(np.max(np.abs(o1 - o2)))
    report["output_max_abs_diff_tfl_vs_onnx"] = maxdiff
    print(f"[*] output max|tfl-onnx| = {maxdiff:.2e}  "
          f"argmax tfl={LABELS[int(np.argmax(o1))]} onnx={LABELS[int(np.argmax(o2))]}")

    # resource baseline
    report["temp_idle_c"] = read_temp()
    report["rss_idle_mb"] = round(psutil.Process().memory_info().rss / 1e6, 1)

    # latency passes (sequential: tflite then onnx)
    print(f"[*] benchmarking TFLite ({args.warmup}w+{args.iters}m) ...", flush=True)
    lat_tfl = bench_one(run_tfl, "tflite", x_tfl, args.iters, args.warmup)
    report["temp_after_tflite_c"] = read_temp()
    report["rss_after_tflite_mb"] = round(psutil.Process().memory_info().rss / 1e6, 1)

    print(f"[*] benchmarking ONNX ({args.warmup}w+{args.iters}m) ...", flush=True)
    lat_onnx = bench_one(run_onnx, "onnx", x_onnx, args.iters, args.warmup)
    report["temp_after_onnx_c"] = read_temp()
    report["rss_after_onnx_mb"] = round(psutil.Process().memory_info().rss / 1e6, 1)

    report["tflite"] = stats(lat_tfl)
    report["onnx"] = stats(lat_onnx)

    # accuracy pass
    print("[*] accuracy pass ...", flush=True)
    report["accuracy"] = {
        "tflite": accuracy(run_tfl, "tflite", args.wavlist, args.acc_limit),
        "onnx": accuracy(run_onnx, "onnx", args.wavlist, args.acc_limit),
    }

    if set_gov and gov_before:
        set_governor(gov_before)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== {args.output} ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
