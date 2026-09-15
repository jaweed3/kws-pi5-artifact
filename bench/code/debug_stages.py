#!/usr/bin/env python3
"""Stage-by-stage diff between numpy and TF.signal MFCC pipeline for ONE wav."""
import sys
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from scipy.fft import dct

from mfcc import (FRAME_LENGTH, FRAME_STEP, FFT_LENGTH, NUM_MEL_BINS,
                  SPECTROGRAM_LENGTH, _periodic_hann, mel_weight_matrix,
                  mfcc_from_audio)

path = sys.argv[1]
rate, data = wavfile.read(path)
audio = data.astype(np.float32) / 32768.0
if audio.ndim > 1:
    audio = audio[:, 0]
print(f"file: {path}  n={len(audio)}")

# ---- TF side ----
a = tf.cast(audio, tf.float32)
a = a / tf.reduce_max(a)
a = tf.pad(a, [[0, 16000 - tf.shape(a)[-1]]])
stfts = tf.signal.stft(a, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP,
                       fft_length=None, window_fn=tf.signal.hann_window)
spec_tf = tf.abs(stfts).numpy()
w_tf = tf.signal.linear_to_mel_weight_matrix(
    NUM_MEL_BINS, spec_tf.shape[-1], 16000, 20.0, 4000.0).numpy()
mel_tf = tf.tensordot(spec_tf, w_tf, 1).numpy()
logmel_tf = np.log(mel_tf + 1e-6)
mfcc_tf = tf.signal.mfccs_from_log_mel_spectrograms(logmel_tf)[..., :10].numpy()

# ---- numpy side ----
an = np.asarray(audio, dtype=np.float32)
mx = an.max()
an2 = an / mx if mx > 0 else an
an2 = np.pad(an2, (0, 16000 - an2.shape[0]), "constant")
window = _periodic_hann(FRAME_LENGTH).astype(np.float32)
starts = np.arange(SPECTROGRAM_LENGTH) * FRAME_STEP
idx = starts[:, None] + np.arange(FRAME_LENGTH)[None, :]
frames = an2[idx] * window
spec_np = np.abs(np.fft.rfft(frames, n=FFT_LENGTH, axis=1))
w_np = mel_weight_matrix()
mel_np = spec_np @ w_np
logmel_np = np.log(mel_np + 1e-6)
mfcc_np = (dct(logmel_np, type=2, norm=None, axis=-1)[:, :10]
           * (1.0 / np.sqrt(2.0 * NUM_MEL_BINS)))

print(f"audio: max_abs_diff={np.max(np.abs(audio - a.numpy())):.3e}")
print(f"normalized audio: max_abs_diff={np.max(np.abs(an2 - a.numpy())):.3e}")
print(f"spec shapes: tf={spec_tf.shape} np={spec_np.shape}")
print(f"spec: max_abs_diff={np.max(np.abs(spec_tf - spec_np)):.3e}  "
      f"rel={np.max(np.abs(spec_tf-spec_np)/(np.abs(spec_tf)+1e-9)):.3e}")
print(f"mel matrix: tf={w_tf.shape} np={w_np.shape} "
      f"max_abs_diff={np.max(np.abs(w_tf - w_np)):.3e}")
print(f"mel: max_abs_diff={np.max(np.abs(mel_tf - mel_np)):.3e}")
print(f"logmel: max_abs_diff={np.max(np.abs(logmel_tf - logmel_np)):.3e}")
print(f"mfcc: max_abs_diff={np.max(np.abs(mfcc_tf - mfcc_np)):.3e}")
print(f"tf mfcc range: {mfcc_tf.min():.3f}..{mfcc_tf.max():.3f}")
print(f"np mfcc range: {mfcc_np.min():.3f}..{mfcc_np.max():.3f}")
