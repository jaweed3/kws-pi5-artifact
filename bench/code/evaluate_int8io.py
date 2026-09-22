#!/usr/bin/env python3
"""Evaluate a true int8-I/O TFLite model (quantize input, dequantize output).

Only needed for kws_ref_model_int8.tflite (the mixed-precision MLPerf
artifact, int8 in/out). All other models use float I/O and evaluate.py.
"""
import argparse
import time

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mfcc import mfcc_from_wav

LABELS = ["Down", "Go", "Left", "No", "Off", "On", "Right",
          "Stop", "Up", "Yes", "Silence", "Unknown"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wavlist", required=True)
    args = ap.parse_args()

    import ai_edge_litert.interpreter as tfl
    interp = tfl.Interpreter(model_path=args.model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    assert inp["dtype"] == np.int8, inp["dtype"]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    print(f"in_scale={in_scale} in_zp={in_zp} out_scale={out_scale} out_zp={out_zp}")

    rows = []
    with open(args.wavlist) as f:
        for line in f:
            line = line.strip()
            if line:
                p, lab = line.split("\t")
                rows.append((p, int(lab)))

    correct = np.zeros(len(LABELS), dtype=int)
    total = np.zeros(len(LABELS), dtype=int)
    t_feat = t_infer = 0.0
    for n, (path, lab) in enumerate(rows, 1):
        t0 = time.perf_counter()
        feat = mfcc_from_wav(path)
        t_feat += time.perf_counter() - t0
        x = np.clip(np.round(feat[None, ...] / in_scale + in_zp),
                    -128, 127).astype(np.int8)
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        logits = (interp.get_tensor(out["index"])[0].astype(np.float32)
                  - out_zp) * out_scale
        t_infer += time.perf_counter() - t0
        pred = int(np.argmax(logits))
        total[lab] += 1
        if pred == lab:
            correct[lab] += 1
        if n % 200 == 0:
            print(f"  [{n}/{len(rows)}] acc_so_far={correct.sum()/max(total.sum(),1):.4f}",
                  flush=True)

    acc = correct.sum() / max(total.sum(), 1)
    print(f"\n=== tflite-int8io :: {args.model} ===")
    print(f"n={total.sum()}  top1={acc:.4f}  ({correct.sum()}/{total.sum()})")
    print(f"feat_time_avg={t_feat/len(rows)*1000:.2f}ms  "
          f"infer_time_avg={t_infer/len(rows)*1000:.3f}ms")
    print("per-class:")
    for i, lab in enumerate(LABELS):
        if total[i]:
            print(f"  {lab:10s} {correct[i]}/{total[i]} = {correct[i]/total[i]:.3f}")


if __name__ == "__main__":
    main()
