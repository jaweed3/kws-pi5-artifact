#!/usr/bin/env python3
"""Build 10-command and full 12-class test lists from Speech Commands v2.

Reads the official testing_list.txt + walks the word dirs (same logic as
train_models.build_dataset), then writes:
  data/testlist_10.tsv  -- 10 command classes only (paper's 4,074-clip subset)
  data/testlist_12.tsv  -- full 12-class task (+ silence + unknown)

Silence: deterministic 1s slices from _background_noise_ (seed 42).
  We use the OFFICIAL MLPerf Tiny convention where possible: the number of
  silence clips in test follows the same generator as training but with a
  disjoint slice range. Unknown: every test clip whose word is not one of the
  10 commands (official testing_list.txt membership decides test inclusion).

Usage:
  python make_testlists.py --data_dir ../data/scv2_full --out_dir ../data
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COMMANDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
LABELS = COMMANDS + ["silence", "unknown"]  # 12 labels, MLPerf Tiny order


def load_wav_audio(path):
    from scipy.io import wavfile
    import numpy as np
    rate, data = wavfile.read(path)
    if rate != 16000:
        raise ValueError(f"{path}: expected 16000 Hz, got {rate}")
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    else:
        audio = data.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, len(audio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data/scv2_full")
    ap.add_argument("--out_dir", default="../data")
    ap.add_argument("--n_silence", type=int, default=1000,
                    help="silence slices in the 12-class list")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    test_paths = set()
    with open(os.path.join(data_dir, "testing_list.txt")) as f:
        for line in f:
            line = line.strip()
            if line:
                test_paths.add(line)

    rows_10, rows_12 = [], []
    n_unknown = 0
    for d in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, d)
        if not os.path.isdir(full) or d.startswith("_") or d == "LICENSE":
            continue
        is_cmd = d in COMMANDS
        label = d if is_cmd else "unknown"
        for f in sorted(os.listdir(full)):
            if not f.endswith(".wav"):
                continue
            rel = f"{d}/{f}"
            if rel not in test_paths:
                continue
            ap_ = os.path.abspath(os.path.join(data_dir, rel))
            lab = LABELS.index(label)
            if is_cmd:
                rows_10.append((ap_, lab))
            else:
                n_unknown += 1
            rows_12.append((ap_, lab))

    # silence slices: disjoint from train/val slice ranges is impossible
    # without the original RNG state, so we use a fixed seed + fixed count
    # and RECORD the slice offsets in a sidecar file for reproducibility.
    from scipy.io import wavfile
    import numpy as np
    noise_dir = os.path.join(data_dir, "_background_noise_")
    wavs = [os.path.join(noise_dir, f) for f in sorted(os.listdir(noise_dir))
            if f.endswith(".wav")]
    auds = []
    for w in wavs:
        rate, data = wavfile.read(w)
        a = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
        if a.ndim > 1:
            a = a[:, 0]
        auds.append(a)
    sr_len = 16000
    offsets = []
    for _ in range(args.n_silence):
        ai = rng.randrange(len(auds))
        a = auds[ai]
        start = rng.randint(0, max(0, len(a) - sr_len)) if len(a) > sr_len else 0
        offsets.append((os.path.basename(wavs[ai]), start))
        rows_12.append((f"_silence_:{os.path.basename(wavs[ai])}:{start}",
                        LABELS.index("silence")))

    # silence needs real wav files for evaluate.py -> materialize to out_dir
    sil_dir = os.path.join(out_dir, "silence_test")
    os.makedirs(sil_dir, exist_ok=True)
    mat_rows = []
    for i, (fn, start) in enumerate(offsets):
        a = auds[[os.path.basename(w) for w in wavs].index(fn)]
        seg = a[start:start + sr_len]
        if len(seg) < sr_len:
            seg = np.pad(seg, (0, sr_len - len(seg)))
        import scipy.io.wavfile as wavf
        p = os.path.join(sil_dir, f"sil_{i:04d}.wav")
        wavf.write(p, 16000, (np.clip(seg, -1, 1) * 32767).astype(np.int16))
        mat_rows.append((p, LABELS.index("silence")))
    rows_12 = [r for r in rows_12 if not r[0].startswith("_silence_:")] + mat_rows

    def write(rows, name):
        p = os.path.join(out_dir, name)
        with open(p, "w") as f:
            for ap_, lab in sorted(rows):
                f.write(f"{ap_}\t{lab}\n")
        print(f"{name}: {len(rows)} clips")
        return p

    write(rows_10, "testlist_10.tsv")
    write(rows_12, "testlist_12.tsv")
    print(f"unknown clips in test: {n_unknown}")
    print("label order:", LABELS)


if __name__ == "__main__":
    main()
