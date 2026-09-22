#!/usr/bin/env python3
"""Thread-scaling sensitivity: TFLite num_threads x ONNX intra_op_num_threads.

Revision experiment for ICAITech R1 ("single-threaded only ... Two- and
four-thread numbers on the top two models would tell me whether the ranking
is stable").

Runs DS-CNN (rebuilt fp32) and CRNN (fp32) under 1/2/4 threads, same protocol
as benchmark_kws.py (1000 warmup + 1000 measured, IQR, 10k bootstrap, seed 42).
Accuracy pass is skipped (thread count does not change numerics); the 1-thread
row must reproduce benchmark_kws_pi.json / bench_crnn_fp32.json within noise.

  python bench_threads.py --models dscnn,crnn --threads 1,2,4 \
      --output results/bench_threads.json

TFLite threads: ai_edge_litert Interpreter NumThreads option. If the installed
LiteRT build ignores the option, the JSON records threads_requested vs the
effective value honestly (see README note) instead of fabricating scaling.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mfcc import mfcc_from_wav  # noqa: E402

MODELS = {
    "dscnn": ("models/kws_ref_model_fp32.tflite",
              "models/kws_ref_model_fp32.onnx"),
    "crnn": ("models/trained/crnn/crnn_fp32.tflite",
             "models/trained/crnn/crnn_fp32.onnx"),
}


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


def load_tflite(path, n_threads):
    import ai_edge_litert.interpreter as tfl
    try:
        interp = tfl.Interpreter(model_path=path, num_threads=n_threads)
        note = f"num_threads={n_threads} accepted"
    except TypeError:
        interp = tfl.Interpreter(model_path=path)
        try:
            interp.set_num_threads(n_threads)
            note = f"set_num_threads({n_threads}) accepted"
        except Exception as e:  # noqa: BLE001
            note = f"thread option unsupported ({e}); single-threaded"
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    def run(x):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(out["index"])
    return run, note


def load_onnx(path, n_threads):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape

    def run(x):
        return sess.run(None, {iname: x})[0]
    return run, (sess, ishape)


def prepare_input(feat, backend, ishape=None):
    x = feat[None, ...].astype(np.float32)
    if backend == "onnx" and ishape and ishape[1] == 1 and ishape[2] != 1:
        x = np.transpose(x, (0, 3, 1, 2))
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
        "n_raw": int(len(samples)), "n_clean": int(len(clean)),
        "n_outliers": int(len(samples) - len(clean)),
        "mean_ms": float(np.mean(clean) * 1000),
        "median_ms": float(np.median(clean) * 1000),
        "p95_ms": float(np.percentile(clean, 95) * 1000),
        "p99_ms": float(np.percentile(clean, 99) * 1000),
        "std_ms": float(np.std(clean) * 1000),
        "ci95_ms": [float(ci[0] * 1000), float(ci[1] * 1000)],
        "fps": float(1.0 / np.mean(clean)),
    }


def bench_one(run_fn, x, iters, warmup):
    for _ in range(warmup):
        run_fn(x)
    lat = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        run_fn(x)
        lat[i] = time.perf_counter() - t0
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dscnn,crnn")
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wav", default=None,
                    help="wav for fixed input (default: first row of testlist)")
    ap.add_argument("--wavlist", default="data/testlist_10.tsv")
    ap.add_argument("--output", default="results/bench_threads.json")
    args = ap.parse_args()

    if args.wav is None and not os.path.exists(args.wavlist):
        for cand in ("data/testlist_10.tsv", "data/testlist.tsv",
                     "../data/testlist.tsv", "testlist.tsv"):
            if os.path.exists(cand):
                print(f"wavlist {args.wavlist} not found, using {cand}",
                      flush=True)
                args.wavlist = cand
                break

    feat = None
    src = args.wav
    if src is None:
        with open(args.wavlist) as f:
            for line in f:
                if line.strip():
                    src = line.split("\t")[0]
                    break
    feat = mfcc_from_wav(src)

    report = {
        "meta": {
            "device": platform.node(), "machine": platform.machine(),
            "python": platform.python_version(), "kernel": platform.release(),
            "governor": governor(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": "warmup+measured, sequential, seed=42, IQR, 10k bootstrap",
        },
        "iters": args.iters, "warmup": args.warmup, "rows": [],
    }

    for mname in [m.strip() for m in args.models.split(",")]:
        tfl_path, onx_path = MODELS[mname]
        for nt in [int(t) for t in args.threads.split(",")]:
            run_tfl, tfl_note = load_tflite(tfl_path, nt)
            run_onx, (_, ishape) = load_onnx(onx_path, nt)
            x_tfl = prepare_input(feat, "tflite")
            x_onx = prepare_input(feat, "onnx", ishape)
            t0 = read_temp()
            lat_tfl = bench_one(run_tfl, x_tfl, args.iters, args.warmup)
            t1 = read_temp()
            lat_onx = bench_one(run_onx, x_onx, args.iters, args.warmup)
            t2 = read_temp()
            s_tfl, s_onx = stats(lat_tfl), stats(lat_onx)
            print(f"[{mname} T={nt}] tflite={s_tfl['mean_ms']:.4f}ms "
                  f"onnx={s_onx['mean_ms']:.4f}ms "
                  f"ratio={s_onx['mean_ms']/s_tfl['mean_ms']:.2f} ({tfl_note})",
                  flush=True)
            report["rows"].append({
                "model": mname, "threads": nt, "tflite_note": tfl_note,
                "temp_before_c": t0, "temp_mid_c": t1, "temp_after_c": t2,
                "tflite": s_tfl, "onnx": s_onx,
                "ratio_onnx_tfl": s_onx["mean_ms"] / s_tfl["mean_ms"],
            })

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
