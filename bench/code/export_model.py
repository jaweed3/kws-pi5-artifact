#!/usr/bin/env python3
"""Re-export a trained .keras model to TFLite fp32 + ONNX (no retraining).

Usage:
  python export_model.py --model lstm --out_dir ../models/trained
"""
import argparse
import os

import numpy as np
import tensorflow as tf

AP = argparse.ArgumentParser()
AP.add_argument("--model", required=True)
AP.add_argument("--out_dir", default="../models/trained")
ARGS = AP.parse_args()

out_dir = os.path.abspath(ARGS.out_dir)
loaded = tf.keras.models.load_model(os.path.join(out_dir, ARGS.model, f"{ARGS.model}.keras"))

# Rebuild with unroll=True for RNN layers: Keras 3 emits TensorList-based
# dynamic loops which the TFLite converter cannot lower. Unrolling keeps the
# exact same weights but produces a static graph the converter accepts.
def make_unrolled(name):
    from tensorflow.keras import layers
    inp = layers.Input(shape=(49, 10, 1), name="input_1")
    if name == "dnn":
        x = layers.Flatten()(inp)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(12, activation="softmax", name="output")(x)
    elif name == "lstm":
        x = layers.Reshape((49, 10))(inp)
        x = layers.LSTM(64, return_sequences=False, unroll=True)(x)
        out = layers.Dense(12, activation="softmax", name="output")(x)
    elif name == "crnn":
        x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Reshape((12, 128))(x)
        x = layers.GRU(64, return_sequences=False, unroll=True)(x)
        out = layers.Dense(12, activation="softmax", name="output")(x)
    else:
        raise ValueError(name)
    return tf.keras.Model(inp, out)

model = make_unrolled(ARGS.model)
model.set_weights(loaded.get_weights())
print(f"rebuilt unrolled model from {ARGS.model}.keras, "
      f"params={model.count_params()}")

# ONNX
import tf2onnx
spec = (tf.TensorSpec((1, 49, 10, 1), tf.float32, name="input_1"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
onnx_path = os.path.join(out_dir, ARGS.model, f"{ARGS.model}_fp32.onnx")
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"onnx: {onnx_path} ({os.path.getsize(onnx_path)} B)")

# TFLite fp32 with tensor-list fallback
import tempfile
sm_dir = tempfile.mkdtemp(prefix="kws_savedmodel_")
model.export(sm_dir)
tfl_path = os.path.join(out_dir, ARGS.model, f"{ARGS.model}_fp32.tflite")
tflite = None
try:
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = []
    tflite = conv.convert()
except Exception as e1:
    print(f"  default converter failed ({type(e1).__name__}); "
          f"retrying with tensor-list lowering disabled")
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = []
    conv._experimental_lower_tensor_list_ops = False
    tflite = conv.convert()
with open(tfl_path, "wb") as f:
    f.write(tflite)
print(f"tflite: {tfl_path} ({os.path.getsize(tfl_path)} B)")
