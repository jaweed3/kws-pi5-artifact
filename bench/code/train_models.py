#!/usr/bin/env python3
"""Train multiple small KWS architectures on Speech Commands v2 (12 labels).

Architectures (all consume the same [49,10,1] MFCC input, output [1,12] softmax):
  - dnn  : dense-only  (~50K params)
  - lstm : LSTM(64)    (~30K params)
  - crnn : conv+GRU    (~90K params)

Preprocessing is numpy-only (code/mfcc.py, verified exact vs tf.signal),
so the trained models are directly comparable with the MLPerf Tiny DS-CNN
reference (kws_ref_model_float32.tflite) under the same benchmark protocol.

Output per model (written to --out_dir/<model>/):
  <model>.keras, <model>_fp32.tflite, <model>_fp32.onnx, history.json
"""
import argparse
import json
import os
import random

import numpy as np

# deterministic data split
random.seed(42)
np.random.seed(42)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from mfcc import mfcc_from_audio, DESIRED_SAMPLES  # noqa: E402

COMMANDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
LABELS = COMMANDS + ["silence", "unknown"]  # 12 labels, MLPerf Tiny order


def _load_wav_audio(path):
    from scipy.io import wavfile
    rate, data = wavfile.read(path)
    if rate != 16000:
        raise ValueError(f"{path}: expected 16000 Hz, got {rate}")
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def _mfcc_of(path):
    return mfcc_from_audio(_load_wav_audio(path))


def silence_slices(noise_dir, n_slices=3000):
    """Random 1s slices from the background noise files."""
    files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir)
             if f.endswith(".wav")]
    audios = [_load_wav_audio(f) for f in files]
    slices = []
    for _ in range(n_slices):
        a = random.choice(audios)
        if a.shape[0] <= DESIRED_SAMPLES:
            start = 0
        else:
            start = random.randint(0, a.shape[0] - DESIRED_SAMPLES)
        slices.append(a[start:start + DESIRED_SAMPLES])
    return slices


def build_dataset(data_dir):
    """Split SCv2 into train/val using the official lists. Returns dicts."""
    val_paths = set()
    with open(os.path.join(data_dir, "validation_list.txt")) as f:
        for line in f:
            val_paths.add(line.strip())
    test_paths = set()
    with open(os.path.join(data_dir, "testing_list.txt")) as f:
        for line in f:
            test_paths.add(line.strip())

    train_files = []   # (relpath, label_idx)
    val_files = []     # (relpath, label_idx)
    noise_dir = os.path.join(data_dir, "_background_noise_")

    for d in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, d)
        if not os.path.isdir(full) or d.startswith("_") or d == "LICENSE":
            continue
        if d in COMMANDS:
            label = d
        else:
            label = "unknown"  # all non-command words
        for f in sorted(os.listdir(full)):
            if not f.endswith(".wav"):
                continue
            rel = f"{d}/{f}"
            if rel in test_paths:
                continue
            if rel in val_paths:
                val_files.append((rel, LABELS.index(label)))
            else:
                train_files.append((rel, LABELS.index(label)))

    # silence: train slices get fixed labels; val slices fixed too
    # (use the first 250 noise slices for val, rest for train)
    noise_slices = silence_slices(noise_dir, n_slices=3250)
    for i, s in enumerate(noise_slices):
        (val_files if i < 250 else train_files).append(
            (f"_silence_{i}", LABELS.index("silence")))

    return train_files, val_files, noise_slices


def precompute(files, audio_cache=None):
    """Map (relpath, label) -> (mfcc [49,10,1], label). Uses cache for silence."""
    out = []
    for rel, lab in files:
        if rel.startswith("_silence_"):
            continue  # handled by caller via audio_cache
        out.append((_mfcc_of(os.path.join(DATA_DIR, rel)), lab))
    return out


def make_model(name):
    import tensorflow as tf
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
        x = layers.LSTM(64, return_sequences=False)(x)
        out = layers.Dense(12, activation="softmax", name="output")(x)
    elif name == "crnn":
        x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Reshape((12, 128))(x)
        x = layers.GRU(64, return_sequences=False)(x)
        out = layers.Dense(12, activation="softmax", name="output")(x)
    else:
        raise ValueError(name)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    global DATA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data/scv2_full")
    ap.add_argument("--out_dir", default="../models/trained")
    ap.add_argument("--model", required=True, choices=["dnn", "lstm", "crnn"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    DATA_DIR = os.path.abspath(args.data_dir)
    out_dir = os.path.join(os.path.abspath(args.out_dir), args.model)
    os.makedirs(out_dir, exist_ok=True)

    import tensorflow as tf
    tf.random.set_seed(42)

    # ---- build dataset
    train_files, val_files, noise_slices = build_dataset(DATA_DIR)
    print(f"train clips: {len(train_files)}  val clips: {len(val_files)}")

    cache = os.path.join(out_dir, "data_cache.npz")
    if os.path.exists(cache):
        print("loading cached features...")
        c = np.load(cache, allow_pickle=True)
        Xtr, ytr, Xva, yva = c["Xtr"], c["ytr"], c["Xva"], c["yva"]
    else:
        print("precomputing MFCC (single-core, ~1 min)...")
        train_no_noise = [(r, l) for r, l in train_files if not r.startswith("_silence_")]
        val_no_noise = [(r, l) for r, l in val_files if not r.startswith("_silence_")]
        # silence features precomputed once
        sil_feats = [mfcc_from_audio(a) for a in noise_slices]

        tr_out = precompute(train_no_noise)
        va_out = precompute(val_no_noise)

        Xtr = np.stack([r[0] for r in tr_out]).astype(np.float32)
        ytr = np.array([r[1] for r in tr_out])
        Xva = np.stack([r[0] for r in va_out]).astype(np.float32)
        yva = np.array([r[1] for r in va_out])

        # append silence (slices 250.. are train; 0..250 val)
        Xtr = np.concatenate([Xtr, np.stack(sil_feats[250:])]).astype(np.float32)
        ytr = np.concatenate([ytr, np.full(len(sil_feats) - 250, LABELS.index("silence"))])
        Xva = np.concatenate([Xva, np.stack(sil_feats[:250])]).astype(np.float32)
        yva = np.concatenate([yva, np.full(250, LABELS.index("silence"))])

        np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)

    print(f"Xtr {Xtr.shape} ytr {ytr.shape} | Xva {Xva.shape} yva {yva.shape}")

    # one-hot + class weights (balanced)
    ytr_oh = tf.keras.utils.to_categorical(ytr, 12)
    yva_oh = tf.keras.utils.to_categorical(yva, 12)
    counts = np.bincount(ytr, minlength=12).astype(np.float64)
    class_weight = {i: counts.max() / c for i, c in enumerate(counts)}
    print("class counts:", counts.astype(int), "\nweights:", {k: round(v, 2) for k, v in class_weight.items()})

    # ---- model
    model = make_model(args.model)
    model.summary()
    n_params = model.count_params()

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-5),
    ]
    hist = model.fit(
        Xtr, ytr_oh,
        validation_data=(Xva, yva_oh),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1,
    )

    model.save(os.path.join(out_dir, f"{args.model}.keras"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({
            "model": args.model,
            "params": int(n_params),
            "train_files": len(train_files),
            "val_files": len(val_files),
            "final_val_acc": float(hist.history["val_accuracy"][-1]),
            "best_val_acc": float(max(hist.history["val_accuracy"])),
            "history": hist.history,
        }, f, indent=2)

    # ---- export ONNX (independent of the TFLite converter bug)
    import tf2onnx
    spec = (tf.TensorSpec((1, 49, 10, 1), tf.float32, name="input_1"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
    onnx_path = os.path.join(out_dir, f"{args.model}_fp32.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # ---- export TFLite fp32 (SavedModel path to dodge the TF2.16 MLIR bug;
    #      LSTM/GRU additionally need tensor-list lowering disabled)
    import tempfile
    sm_dir = tempfile.mkdtemp(prefix="kws_savedmodel_")
    model.export(sm_dir)
    tfl_path = os.path.join(out_dir, f"{args.model}_fp32.tflite")
    tflite = None
    try:
        conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
        conv.optimizations = []
        tflite = conv.convert()
    except Exception as e1:
        print(f"  tflite default converter failed ({type(e1).__name__}); "
              f"retrying with tensor-list lowering disabled")
        conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
        conv.optimizations = []
        conv._experimental_lower_tensor_list_ops = False
        tflite = conv.convert()
    with open(tfl_path, "wb") as f:
        f.write(tflite)

    print(f"OK: {args.model} params={n_params} best_val_acc={max(hist.history['val_accuracy']):.4f}")
    print(f"    tflite={tfl_path} ({os.path.getsize(tfl_path)} B)")
    print(f"    onnx={onnx_path} ({os.path.getsize(onnx_path)} B)")




if __name__ == "__main__":
    main()
