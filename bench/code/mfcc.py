#!/usr/bin/env python3
"""TF-exact MFCC frontend for MLPerf Tiny KWS reference model (Speech Commands v2).

Replicates, in pure numpy, the preprocessing pipeline of
mlcommons/tiny benchmark/training/keyword_spotting/get_dataset.py
(which uses tf.signal). Any deviation here silently destroys accuracy,
so the constants below were copied from the reference implementation:

  sample_rate       = 16000
  clip_duration_ms  = 1000  -> desired_samples = 16000
  window_size_ms    = 30    -> frame_length   = 480 samples
  window_stride_ms  = 20    -> frame_step     = 320 samples
  fft_length        = None  -> next_pow2(480) = 512  -> 257 bins
  window_fn         = tf.signal.hann_window (PERIODIC Hann)
  num_mel_bins      = 40, lower_edge 20 Hz, upper_edge 4000 Hz
  mel scale         = Slaney: mel(f) = 1127 * ln(1 + f/700)
  mel matrix        = triangular, peak 1.0, NOT normalized, DC bin zeroed
  log               = ln(mel + 1e-6)
  DCT               = type-II, norm=None (unscaled), first 10 coeffs
  output            = [49, 10, 1]  (spectrogram_length x dct_coeffs x 1)

Input normalization (inference path, no training distortions):
  audio (float, [-1,1]) / reduce_max(audio)  ->  per-clip peak normalize
  pad to 16000 with zeros if shorter
"""
import numpy as np
from scipy.fft import dct
from scipy.io import wavfile

SAMPLE_RATE = 16000
DESIRED_SAMPLES = 16000
FRAME_LENGTH = 480
FRAME_STEP = 320
FFT_LENGTH = 512
NUM_SPECTROGRAM_BINS = FFT_LENGTH // 2 + 1  # 257
NUM_MEL_BINS = 40
LOWER_EDGE_HZ = 20.0
UPPER_EDGE_HZ = 4000.0
NUM_CEPSTRAL = 10
SPECTROGRAM_LENGTH = 1 + (DESIRED_SAMPLES - FRAME_LENGTH) // FRAME_STEP  # 49


def _hertz_to_mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


def mel_weight_matrix(num_bins=NUM_MEL_BINS, num_bins_spec=NUM_SPECTROGRAM_BINS,
                      sr=SAMPLE_RATE, lo=LOWER_EDGE_HZ, hi=UPPER_EDGE_HZ):
    """Replicates tf.signal.linear_to_mel_weight_matrix exactly (TF 2.16)."""
    nyquist = sr / 2.0
    # linear frequencies for bins 1..256 (DC bin excluded, zeroed later)
    lin = np.linspace(0.0, nyquist, num_bins_spec)[1:]
    bins_mel = _hertz_to_mel(lin)[:, None]                       # [256, 1]
    # TF: linspace in MEL space (uniform mel), NOT mel of uniform Hz
    edges = np.linspace(_hertz_to_mel(lo), _hertz_to_mel(hi), num_bins + 2)
    lower = edges[:-2][None, :]
    center = edges[1:-1][None, :]
    upper = edges[2:][None, :]
    lower_slopes = (bins_mel - lower) / (center - lower)
    upper_slopes = (upper - bins_mel) / (upper - center)
    w = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))  # [256, 40]
    return np.vstack([np.zeros((1, num_bins)), w])               # [257, 40]


_MEL_MATRIX = mel_weight_matrix()


def _periodic_hann(n):
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))


def mfcc_from_audio(audio):
    """audio: float32 1-D array (any length <= 16000). Returns [49,10,1] float32."""
    a = np.asarray(audio, dtype=np.float32)
    # per-clip peak normalization (divide by reduce_max, TF semantics)
    mx = a.max()
    if mx > 0:
        a = a / mx
    # pad to desired length
    if a.shape[0] < DESIRED_SAMPLES:
        a = np.pad(a, (0, DESIRED_SAMPLES - a.shape[0]), "constant")
    # framed STFT with periodic Hann, zero-padded FFT to 512
    window = _periodic_hann(FRAME_LENGTH).astype(np.float32)
    starts = np.arange(SPECTROGRAM_LENGTH) * FRAME_STEP
    idx = starts[:, None] + np.arange(FRAME_LENGTH)[None, :]
    frames = a[idx] * window                                  # [49, 480]
    spec = np.abs(np.fft.rfft(frames, n=FFT_LENGTH, axis=1))  # [49, 257]
    mel = spec @ _MEL_MATRIX                                  # [49, 40]
    log_mel = np.log(mel + 1e-6)
    # tf.signal.mfccs_from_log_mel_spectrograms = dct(type=2, norm=None)
    #   * rsqrt(2 * num_mel_bins)   <- HTK "almost orthogonal" scaling
    mfccs = (dct(log_mel, type=2, norm=None, axis=-1)[:, :NUM_CEPSTRAL]
             * (1.0 / np.sqrt(2.0 * NUM_MEL_BINS)))
    return mfccs.astype(np.float32).reshape(SPECTROGRAM_LENGTH, NUM_CEPSTRAL, 1)


def mfcc_from_wav(path):
    """Load a Speech Commands v2 wav (16k mono) and return its MFCC [49,10,1]."""
    rate, data = wavfile.read(path)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate}")
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    return mfcc_from_audio(audio)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        f = mfcc_from_wav(p)
        print(f"{p}: shape={f.shape} min={f.min():.3f} max={f.max():.3f} "
              f"mean={f.mean():.3f} finite={np.isfinite(f).all()}")
