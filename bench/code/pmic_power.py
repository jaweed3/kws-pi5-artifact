#!/usr/bin/env python3
"""PMIC-based power measurement for KWS inference on Raspberry Pi 5.

Reads vcgencmd pmic_read_adc to sum V*I across on-board rails.
Method: aggregate — P_idle baseline (30s), then P_load during continuous
inference loop (60s) with PMIC sampler at ~50ms intervals.

Energy per inference = (P_load - P_idle) * mean_latency_ms / 1000  [mJ]

Usage (from bench/code, via drone_venv):
  python bench/code/pmic_power.py --model dscnn --runtime tflite --iters 10000
  python bench/code/pmic_power.py --model crnn --runtime onnx --iters 10000
  python bench/code/pmic_power.py --all          # run all configs sequentially
"""
import argparse, json, os, platform, subprocess, sys, time, re, threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mfcc import mfcc_from_wav

# ── CONFIG ────────────────────────────────────────────────────────
MODELS = {
    "dscnn": {
        "tflite": "models/kws_ref_model_fp32.tflite",
        "onnx":   "models/kws_ref_model_fp32.onnx",
    },
    "crnn": {
        "tflite": "models/trained/crnn/crnn_fp32.tflite",
        "onnx":   "models/trained/crnn/crnn_fp32.onnx",
    },
    "dnn": {
        "tflite": "models/trained/dnn/dnn_fp32.tflite",
        "onnx":   "models/trained/dnn/dnn_fp32.onnx",
    },
    "lstm": {
        "tflite": "models/trained/lstm/lstm_fp32.tflite",
        "onnx":   "models/trained/lstm/lstm_fp32.onnx",
    },
}

SAMPLES_PER_SEC = 20          # PMIC read interval ~50ms
IDLE_SECONDS = 30             # baseline P_idle duration
LOAD_SECONDS = 60             # P_load during inference loop
WARMUP_PASSES = 500           # warmup before timed loop
WAVLIST = "data/testlist.tsv"
SEED = 42

# ── PMIC PARSER ───────────────────────────────────────────────────
def read_pmic_watts():
    """Read PMIC ADC and return total on-board power in Watts."""
    try:
        out = subprocess.run(
            ["vcgencmd", "pmic_read_adc"],
            capture_output=True, text=True, timeout=3
        ).stdout
        # Parse: rails end with _A (current) or _V (voltage)
        # Pair them by base name: NAME_A + NAME_V = one rail
        current_rails = {}  # base_name -> amps
        voltage_rails = {}  # base_name -> volts
        for m in re.finditer(r"(\S+?)\s+(current|volt)\(\d+\)=([0-9.]+)[AV]", out):
            raw_name, typ, val = m.group(1), m.group(2), float(m.group(3))
            if raw_name.endswith("_A") and typ == "current":
                base = raw_name[:-2]
                current_rails[base] = val
            elif raw_name.endswith("_V") and typ == "volt":
                base = raw_name[:-2]
                voltage_rails[base] = val

        total_w = 0.0
        matched = []
        for base in sorted(set(current_rails.keys()) & set(voltage_rails.keys())):
            w = current_rails[base] * voltage_rails[base]
            total_w += w
            matched.append((base, current_rails[base], voltage_rails[base], w))
        return total_w, matched
    except Exception:
        return None

def read_temp():
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=2).stdout
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None

def governor():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except Exception:
        return None

# ── MODEL LOADERS ─────────────────────────────────────────────────
def load_tflite(path):
    import ai_edge_litert.interpreter as tfl
    try:
        interp = tfl.Interpreter(model_path=path, num_threads=1)
    except TypeError:
        interp = tfl.Interpreter(model_path=path)
        try: interp.set_num_threads(1)
        except: pass
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    def run(x):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(out["index"])
    return run

def load_onnx(path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    def run(x):
        return sess.run(None, {iname: x})[0]
    return run

def prep(feat, backend, ishape=None):
    x = feat[None, ...].astype(np.float32)
    if backend == "onnx" and ishape and ishape[1] == 1 and ishape[2] != 1:
        x = np.transpose(x, (0, 3, 1, 2))
    return x

# ── PMIC SAMPLER THREAD ──────────────────────────────────────────
class PMICSampler:
    """Background thread that samples PMIC power at fixed intervals."""
    def __init__(self, interval_ms=50):
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        self.samples.clear()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            result = read_pmic_watts()
            w = result[0] if isinstance(result, tuple) else result
            if w is not None:
                self.samples.append((time.time(), w))
            # sleep remainder of interval
            time.sleep(max(0, self.interval - 0.005))

    def mean_watts(self):
        if not self.samples:
            return 0.0
        return float(np.mean([s[1] for s in self.samples]))

    def summary(self):
        watts = [s[1] for s in self.samples]
        return {
            "n_samples": len(watts),
            "mean_w": float(np.mean(watts)) if watts else 0.0,
            "std_w": float(np.std(watts)) if watts else 0.0,
            "min_w": float(np.min(watts)) if watts else 0.0,
            "max_w": float(np.max(watts)) if watts else 0.0,
        }

# ── LATENCY STATS ────────────────────────────────────────────────
def stats_from_ns(samples_ns):
    s = np.asarray(samples_ns, dtype=np.float64)
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    clean = s[(s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)]
    return {
        "n_raw": int(len(s)),
        "n_clean": int(len(clean)),
        "n_outliers": int(len(s) - len(clean)),
        "mean_ms": float(np.mean(clean) * 1e-6),
        "median_ms": float(np.median(clean) * 1e-6),
        "std_ms": float(np.std(clean) * 1e-6),
        "p95_ms": float(np.percentile(clean, 95) * 1e-6),
        "p99_ms": float(np.percentile(clean, 99) * 1e-6),
    }

# ── MAIN MEASUREMENT ─────────────────────────────────────────────
def measure_config(model_name, runtime, wav_paths, feat_cache, iters):
    model_path = MODELS[model_name][runtime]
    if not os.path.exists(model_path):
        print(f"  [SKIP] model not found: {model_path}")
        return None

    print(f"  Loading {model_name}/{runtime}: {model_path}")
    if runtime == "tflite":
        run_fn = load_tflite(model_path)
    else:
        run_fn = load_onnx(model_path)

    # Warmup
    print(f"  Warmup ({WARMUP_PASSES} passes)...")
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(wav_paths), size=WARMUP_PASSES, replace=True)
    for i in idxs:
        run_fn(prep(feat_cache[i], runtime))

    # ── PMIC BASELINE (idle, no inference) ──
    print(f"  Measuring idle baseline ({IDLE_SECONDS}s)...")
    sampler = PMICSampler(interval_ms=50)
    sampler.start()
    time.sleep(IDLE_SECONDS)
    sampler.stop()
    p_idle = sampler.mean_watts()
    idle_summary = sampler.summary()
    print(f"  P_idle = {p_idle:.4f} W ({idle_summary['n_samples']} samples)")

    # ── PMIC LOAD (continuous inference) ──
    print(f"  Measuring load ({iters} passes, ~{LOAD_SECONDS}s)...")
    sampler = PMICSampler(interval_ms=50)
    sampler.start()

    latency_ns = []
    t_start = time.time()
    for _ in range(iters):
        idx = rng.integers(0, len(wav_paths))
        t0 = time.perf_counter_ns()
        run_fn(prep(feat_cache[idx], runtime))
        t1 = time.perf_counter_ns()
        latency_ns.append(t1 - t0)
    elapsed = time.time() - t_start

    sampler.stop()
    p_load = sampler.mean_watts()
    load_summary = sampler.summary()
    print(f"  P_load = {p_load:.4f} W ({load_summary['n_samples']} samples)")

    lat = stats_from_ns(latency_ns)
    p_delta = max(0.0, p_load - p_idle)
    # E_inference = P_delta * mean_latency [mJ]
    # because Joule = W * s, and mean_ms * 1e-3 = seconds
    # mJ = W * ms (conveniently, W * ms = mJ when ms in 10^-3 and we want 10^-3)
    e_per_inf_mj = p_delta * lat["mean_ms"]
    # E_total = P_delta * elapsed
    e_total_j = p_delta * elapsed

    result = {
        "model": model_name,
        "runtime": runtime,
        "threads": 1,
        "params": {"iters": iters, "warmup": WARMUP_PASSES, "seed": SEED},
        "device": platform.node(),
        "platform": platform.platform(),
        "governor": governor(),
        "temp_idle_c": None,
        "power": {
            "method": "pmic_read_adc (on-board PMIC, excludes USB peripherals)",
            "p_idle_w": round(p_idle, 4),
            "p_load_w": round(p_load, 4),
            "p_delta_w": round(p_delta, 4),
            "e_per_inference_mj": round(e_per_inf_mj, 6),
            "e_total_j": round(e_total_j, 6),
            "elapsed_s": round(elapsed, 2),
            "idle_samples": idle_summary["n_samples"],
            "load_samples": load_summary["n_samples"],
            "idle_summary": idle_summary,
            "load_summary": load_summary,
        },
        "latency": lat,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result

# ── CLI ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="PMIC power measurement")
    ap.add_argument("--model", choices=MODELS.keys(), help="single model")
    ap.add_argument("--runtime", choices=["tflite", "onnx"], help="single runtime")
    ap.add_argument("--all", action="store_true", help="run all model×runtime")
    ap.add_argument("--iters", type=int, default=10000, help="measured passes")
    ap.add_argument("--output", default=None, help="output JSON path")
    args = ap.parse_args()

    # chdir to bench/ so paths in testlist.tsv resolve correctly
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    # Load test data
    print("Loading test data...")
    wav_paths = []
    labels = []
    with open(WAVLIST) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                wav_paths.append(parts[0])
                labels.append(int(parts[1]))

    # Cache features
    print(f"Precomputing MFCC for {len(wav_paths)} clips...")
    feat_cache = []
    for p in wav_paths:
        feat_cache.append(mfcc_from_wav(p))
    print("Done.\n")

    configs = []
    if args.all:
        for m in MODELS:
            for r in ["tflite", "onnx"]:
                configs.append((m, r))
    elif args.model and args.runtime:
        configs.append((args.model, args.runtime))
    else:
        print("Specify --model MODEL --runtime RUNTIME, or --all")
        sys.exit(1)

    results = []
    for model_name, runtime in configs:
        print(f"\n{'='*50}")
        print(f"CONFIG: {model_name} / {runtime}")
        print(f"{'='*50}")
        r = measure_config(model_name, runtime, wav_paths, feat_cache, args.iters)
        if r:
            results.append(r)
            # Print summary
            lat = r["latency"]
            pw = r["power"]
            print(f"\n  RESULT:")
            print(f"    mean latency: {lat['mean_ms']:.4f} ms (±{lat['std_ms']:.4f})")
            print(f"    P_idle:  {pw['p_idle_w']:.4f} W")
            print(f"    P_load:  {pw['p_load_w']:.4f} W")
            print(f"    P_delta: {pw['p_delta_w']:.4f} W")
            print(f"    Energy:  {pw['e_per_inference_mj']:.4f} mJ/inf")
            print(f"    Total:   {pw['e_total_j']:.6f} J over {pw['elapsed_s']:.1f}s")

    # Save
    if results:
        ts = time.strftime("%Y%m%d_%H%M")
        outpath = args.output or f"results/power_pmic_{ts}.json"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {outpath}")

if __name__ == "__main__":
    main()
