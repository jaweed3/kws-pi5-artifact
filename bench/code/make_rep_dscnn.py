#!/usr/bin/env python3
"""Make a small representative MFCC set for DS-CNN int8 calibration (runs on Mac/PC).

Samples --n_rep training clips (seed 42), computes MFCC with the same numpy
front end, saves data/dscnn_rep.npz (~2MB). Ship the npz to the Pi5; the
quantizer there needs no dataset, no TF training env, no 5.6GB download.

  python make_rep_dscnn.py --data_dir ../data/scv2_full --n_rep 1000
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COMMANDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data/scv2_full")
    ap.add_argument("--out", default="../data/dscnn_rep.npz")
    ap.add_argument("--n_rep", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from mfcc import mfcc_from_audio
    from scipy.io import wavfile

    rng = random.Random(args.seed)
    data_dir = os.path.abspath(args.data_dir)

    test_paths, val_paths = set(), set()
    for name, s in (("testing_list.txt", test_paths),
                    ("validation_list.txt", val_paths)):
        with open(os.path.join(data_dir, name)) as f:
            for line in f:
                line = line.strip()
                if line:
                    s.add(line)

    cands = []
    for d in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, d)
        if not os.path.isdir(full) or d.startswith("_") or d == "LICENSE":
            continue
        for f in sorted(os.listdir(full)):
            if not f.endswith(".wav"):
                continue
            rel = f"{d}/{f}"
            if rel in test_paths or rel in val_paths:
                continue
            cands.append(os.path.join(data_dir, rel))
    rng.shuffle(cands)
    cands = cands[:args.n_rep]

    feats = []
    for p in cands:
        rate, data = wavfile.read(p)
        assert rate == 16000, p
        audio = (data.astype(np.float32) / 32768.0 if data.dtype == np.int16
                 else data.astype(np.float32))
        if audio.ndim > 1:
            audio = audio[:, 0]
        feats.append(mfcc_from_audio(audio).astype(np.float32))
    arr = np.stack(feats)  # [n,49,10,1]
    assert arr.shape[1:] == (49, 10, 1), arr.shape
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, mfcc=arr)
    print(f"wrote {args.out}: {arr.shape} "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
