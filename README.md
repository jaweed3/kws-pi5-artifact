# KWS Benchmark — TFLite vs ONNX Runtime on Raspberry Pi 5

Benchmark keyword-spotting architectures (DS-CNN, DNN, LSTM, CRNN) across
TensorFlow Lite (float32 & int8) dan ONNX Runtime (float32) di Raspberry Pi 5.

Paper: *Benchmarking Keyword Spotting Architectures Across TensorFlow Lite and ONNX Runtime on a Raspberry Pi 5*
(ICAITech 2026, submitted — double-blind, repo private).

## Struktur

| Folder | Isi |
|---|---|
| `paper/` | Source LaTeX (`main.tex`), PDF final (`main.pdf`), `refs.bib`, figure scripts (`gen_figure_*.py`), review manuscript `.doc` (EDAS) |
| `bench/code/` | Pipeline: `train_models.py` → `export_model.py` → `quantize_model.py` → `benchmark_kws.py` |
| `bench/results/` | Raw benchmark JSON (latency fp32/int8) |
| `bench/models/` | Model artifacts: MLPerf reference + trained (keras/tflite/onnx) per arsitektur |
| `literature/` | Literaur open-access (PDF) + skrip download |
| `RESEARCH_BRIEF.md` | Ringkasan riset |

## Catatan penting

- Dataset Speech Commands v2 (5.6GB) dan `data_cache.npz` (cache MFCC, ~166MB/model)
  TIDAK di-commit — regenerable lewat `bench/` scripts.
- `literature/paywalled/` (18MB) sengaja tidak di-commit (lisensi) — ada lokal.
- Jangan di-public sebelum paper accepted.

## Status

- ICAITech 2026: submitted (review via EDAS, deadline 12 Aug 2026, keputusan 15 Sep)
- MLCIPR: fallback
