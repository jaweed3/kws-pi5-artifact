#!/usr/bin/env python3
"""Top-1 accuracy of a KWS model over a wav list, using the numpy MFCC frontend.

Validates the whole preprocessing + model pipeline. Runs the SAME wavs through
either runtime and reports per-class and overall top-1 accuracy.

Usage:
  python evaluate.py --model models/kws_ref_model.onnx --backend onnx \
      --wavlist data/testlist.tsv
  python evaluate.py --model models/kws_ref_model_float32.tflite --backend tflite \
      --wavlist data/testlist.tsv

--wavlist: TSV lines of  <path> <label_idx>
"""
import argparse
import time

import numpy as np

from mfcc import mfcc_from_wav

LABELS = ["Down", "Go", "Left", "No", "Off", "On", "Right",
          "Stop", "Up", "Yes", "Silence", "Unknown"]


def load_model(path, backend):
    if backend == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        ishape = sess.get_inputs()[0].shape  # [1,49,10,1] NHWC or [1,1,49,10] NCHW
        def infer(feat):
            x = feat[None, ...]  # [49,10,1] -> [1,49,10,1]
            if ishape[1] == 1 and ishape[2] != 1:
                x = np.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW [1,1,49,10]
            return sess.run(None, {iname: x.astype(np.float32)})[0]
        return infer
    elif backend == "tflite":
        import ai_edge_litert.interpreter as tfl
        interp = tfl.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        def infer(feat):
            x = feat[None, ...]  # [49,10,1] -> [1,49,10,1] (NHWC)
            interp.set_tensor(inp["index"], x.astype(np.float32))
            interp.invoke()
            return interp.get_tensor(out["index"])
        return infer
    raise ValueError(f"unknown backend {backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["onnx", "tflite"], required=True)
    ap.add_argument("--wavlist", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.wavlist) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, lab = line.split("\t")
            rows.append((path, int(lab)))
    if args.limit:
        rows = rows[: args.limit]

    infer = load_model(args.model, args.backend)

    t_feat = t_infer = 0.0
    correct = np.zeros(len(LABELS), dtype=int)
    total = np.zeros(len(LABELS), dtype=int)
    n = 0
    for path, lab in rows:
        t0 = time.perf_counter()
        feat = mfcc_from_wav(path)
        t_feat += time.perf_counter() - t0
        t0 = time.perf_counter()
        logits = infer(feat)[0]
        t_infer += time.perf_counter() - t0
        pred = int(np.argmax(logits))
        total[lab] += 1
        if pred == lab:
            correct[lab] += 1
        n += 1
        if n % 200 == 0:
            print(f"  [{n}/{len(rows)}] acc_so_far={correct.sum()/max(total.sum(),1):.4f}", flush=True)

    acc = correct.sum() / max(total.sum(), 1)
    print(f"\n=== {args.backend} :: {args.model} ===")
    print(f"n={total.sum()}  top1={acc:.4f}  ({correct.sum()}/{total.sum()})")
    print(f"feat_time_avg={t_feat/n*1000:.2f}ms  infer_time_avg={t_infer/n*1000:.3f}ms")
    print("per-class:")
    for i, lab in enumerate(LABELS):
        if total[i]:
            print(f"  {lab:10s} {correct[i]}/{total[i]} = {correct[i]/total[i]:.3f}")


if __name__ == "__main__":
    main()
