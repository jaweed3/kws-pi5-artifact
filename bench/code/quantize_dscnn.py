#!/usr/bin/env python3
"""Quantize the rebuilt true-float32 DS-CNN to full int8 (runs on the Pi5).

Revision experiment for ICAITech R1/R3: the submitted paper's DS-CNN "int8"
cell is the original mixed-precision MLPerf artifact (int8 conv kernels,
float32 remainder), NOT a full-int8 quantization like the other three rows.
This script closes that gap with no retraining: it rebuilds the verified
fp32 Keras model (rebuild_model.build_dscnn_keras), calibrates on
data/dscnn_rep.npz (built on Mac/PC by make_rep_dscnn.py, seed 42), and writes
models/kws_ref_model_fullint8.tflite.

After quantizing, run the standard protocol on it:
  python benchmark_kws.py --tflite models/kws_ref_model_fullint8.tflite \
      --onnx models/kws_ref_model_fp32.onnx --tag dscnn_fullint8 \
      --output results/bench_dscnn_fullint8.json
(the onnx column is the fp32 reference; only the tflite column is new data)
and accuracy on both lists:
  python evaluate.py --model models/kws_ref_model_fullint8.tflite \
      --backend tflite --wavlist data/testlist_10.tsv
  python evaluate.py --model models/kws_ref_model_fullint8.tflite \
      --backend tflite --wavlist data/testlist_12.tsv
"""
import argparse
import os

import numpy as np

from rebuild_model import build_dscnn_keras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/kws_ref_model_float32.tflite",
                    help="mixed-precision flatbuffer to rebuild weights from")
    ap.add_argument("--rep", default="../data/dscnn_rep.npz")
    ap.add_argument("--out", default="../models/kws_ref_model_fullint8.tflite")
    args = ap.parse_args()

    import tensorflow as tf

    rep = np.load(os.path.abspath(args.rep))["mfcc"].astype(np.float32)
    print(f"rep set: {rep.shape}")

    model = build_dscnn_keras(os.path.abspath(args.src))

    def rep_gen():
        for x in rep:
            yield [x[None, ...]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32  # keep float I/O, int8 internal
    conv.inference_output_type = tf.float32
    tflite_int8 = conv.convert()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(tflite_int8)
    print(f"wrote {out} ({len(tflite_int8)} bytes)")


if __name__ == "__main__":
    main()
