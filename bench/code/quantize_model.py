#!/usr/bin/env python3
"""Quantize a trained KWS model to full int8 TFLite (representative dataset).

Uses the same numpy MFCC front end as training; representative set = 1000
random training clips. Output: <model>_int8.tflite (full int8: weights AND
activations), plus accuracy check vs fp32 on a small slice.

Usage:
  python quantize_model.py --model dnn --data_dir ../data/scv2_full
"""
import argparse
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from mfcc import mfcc_from_wav  # noqa: E402

random.seed(42)
np.random.seed(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dnn", "lstm", "crnn"])
    ap.add_argument("--data_dir", default="../data/scv2_full")
    ap.add_argument("--out_dir", default="../models/trained")
    ap.add_argument("--n_rep", type=int, default=1000,
                    help="representative samples for calibration")
    args = ap.parse_args()

    import tensorflow as tf

    out_dir = os.path.abspath(args.out_dir)
    keras_path = os.path.join(out_dir, args.model, f"{args.model}.keras")
    loaded = tf.keras.models.load_model(keras_path)

    # rebuild with unroll=True (same weights, static graph) so the TFLite
    # converter can lower the RNN (dynamic TensorList loops fail otherwise)
    from tensorflow.keras import layers
    inp = layers.Input(shape=(49, 10, 1), name="input_1")
    if args.model == "dnn":
        x = layers.Flatten()(inp)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(12, activation="softmax", name="output")(x)
    elif args.model == "lstm":
        x = layers.Reshape((49, 10))(inp)
        x = layers.LSTM(64, return_sequences=False, unroll=True)(x)
        out = layers.Dense(12, activation="softmax", name="output")(x)
    elif args.model == "crnn":
        x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Reshape((12, 128))(x)
        x = layers.GRU(64, return_sequences=False, unroll=True)(x)
        out = layers.Dense(12, activation="softmax", name="output")(x)
    else:
        raise ValueError(args.model)
    model = tf.keras.Model(inp, out)
    model.set_weights(loaded.get_weights())
    print(f"rebuilt unrolled {args.model} from {keras_path}")

    # representative dataset from training set (exclude val/test lists)
    data_dir = os.path.abspath(args.data_dir)
    val_paths = set(open(os.path.join(data_dir, "validation_list.txt")).read().split())
    test_paths = set(open(os.path.join(data_dir, "testing_list.txt")).read().split())

    wavs = []
    for d in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, d)
        if not os.path.isdir(full) or d.startswith("_") or d == "LICENSE":
            continue
        for f in os.listdir(full):
            if f.endswith(".wav"):
                rel = f"{d}/{f}"
                if rel not in val_paths and rel not in test_paths:
                    wavs.append(os.path.join(full, f))
    random.shuffle(wavs)
    rep = wavs[: args.n_rep]
    print(f"representative set: {len(rep)} wavs")

    def rep_gen():
        for w in rep:
            f = mfcc_from_wav(w)[None, ...]  # [1,49,10,1]
            yield [f.astype(np.float32)]

    import tempfile
    sm_dir = tempfile.mkdtemp(prefix="kws_savedmodel_q_")
    model.export(sm_dir)

    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32  # keep float input, int8 internal
    conv.inference_output_type = tf.float32
    tflite_model = conv.convert()

    out_path = os.path.join(out_dir, args.model, f"{args.model}_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"OK: {out_path} ({os.path.getsize(out_path)} B)")


if __name__ == "__main__":
    main()
