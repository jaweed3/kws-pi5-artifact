#!/usr/bin/env python3
"""Diff numpy MFCC vs exact TF.signal reference, on the same wavs.

Usage: python diff_tf_mfcc.py <wav> [<wav> ...]
Prints max abs diff per file; if < 1e-4 the numpy frontend is exact.
"""
import sys
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

from mfcc import mfcc_from_audio


def tf_reference_mfcc(audio):
    a = tf.cast(audio, tf.float32)
    a = a / tf.reduce_max(a)
    a = tf.pad(a, [[0, 16000 - tf.shape(a)[-1]]])
    stfts = tf.signal.stft(a, frame_length=480, frame_step=320,
                           fft_length=None, window_fn=tf.signal.hann_window)
    spectrograms = tf.abs(stfts)
    nbins = stfts.shape[-1]
    w = tf.signal.linear_to_mel_weight_matrix(40, nbins, 16000, 20.0, 4000.0)
    mel = tf.tensordot(spectrograms, w, 1)
    log_mel = tf.math.log(mel + 1e-6)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :10]
    return mfcc.numpy()  # [49, 10]


def main():
    worst = 0.0
    for path in sys.argv[1:]:
        rate, data = wavfile.read(path)
        audio = data.astype(np.float32) / 32768.0
        if audio.ndim > 1:
            audio = audio[:, 0]
        ref = tf_reference_mfcc(audio)
        mine = mfcc_from_audio(audio)[..., 0]  # [49,10]
        d = float(np.max(np.abs(ref - mine)))
        rel = float(np.max(np.abs(ref - mine) / (np.abs(ref) + 1e-9)))
        worst = max(worst, d)
        print(f"{path.split('/')[-1]:40s} max_abs_diff={d:.3e} max_rel={rel:.3e}")
    print(f"\nWORST max_abs_diff = {worst:.3e}")


if __name__ == "__main__":
    main()
