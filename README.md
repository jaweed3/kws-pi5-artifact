# KWS Benchmark Artifact — TFLite vs ONNX Runtime on Raspberry Pi 5

Artifact for: *Benchmarking Keyword Spotting Architectures Across TensorFlow
Lite and ONNX Runtime on a Raspberry Pi 5* (ICAITech 2026).

Four keyword-spotting architectures (DS-CNN / MLPerf Tiny reference, dense,
LSTM, CNN-GRU hybrid) × three runtime configs (float32 TFLite, int8 TFLite,
float32 ONNX Runtime) on a Raspberry Pi 5, under one controlled protocol.

## What's inside

| Path | Contents |
|---|---|
| `paper/` | LaTeX source (`main.tex`), compiled PDF, `refs.bib`, figure scripts |
| `bench/code/` | Full pipeline: train → export → quantize → benchmark → evaluate |
| `bench/models/` | All model artifacts (MLPerf reference + trained `.keras`/`.tflite`/`.onnx`) |
| `bench/results/` | Raw benchmark JSONs (latency, accuracy, protocol metadata) |
| `SESSION_RUNBOOK.md` | Step-by-step revision experiment session guide |

## Reproduce (on a Raspberry Pi 5)

Requirements: Python 3.13, TensorFlow 2.16, `ai-edge-litert`, `onnxruntime`,
`tf2onnx`, `scipy`, `numpy`, `psutil`. CPU governor pinned to `performance`
(the scripts do this automatically via `sudo tee`).

Dataset: Google Speech Commands v2 ([CC BY 4.0](https://arxiv.org/abs/1804.03209),
~5.6 GB, **not** included — download to `bench/data/scv2_full/`):

```bash
cd bench/code
python train_models.py --model dnn --data_dir ../data/scv2_full    # repeat: lstm, crnn
python rebuild_model.py                                            # true-float32 DS-CNN from MLPerf artifact
python quantize_model.py --model dnn --data_dir ../data/scv2_full  # repeat: lstm, crnn
python quantize_dscnn.py --rep ../data/dscnn_rep.npz               # full-int8 DS-CNN
python benchmark_kws.py --tflite ../models/trained/dnn/dnn_fp32.tflite \
    --onnx ../models/trained/dnn/dnn_fp32.onnx --tag dnn_fp32 \
    --output ../results/bench_dnn_fp32.json
python evaluate.py --model ../models/trained/dnn/dnn_fp32.tflite \
    --backend tflite --wavlist ../data/testlist_10.tsv
python bench_threads.py --models dscnn,crnn --threads 1,2,4       # thread scaling
python make_testlists.py --data_dir ../data/scv2_full             # 10-cmd + 12-class lists
```

Protocol (every run): 1,000 warmup + 1,000 measured passes, sequential,
single thread unless stated, features precomputed, IQR outlier removal,
10,000-resample bootstrap 95% CI, seed 42.

## Verify without hardware

`bench/results/*.json` contain the raw numbers behind every table in the
paper (means, medians, p95/p99, CIs, outlier counts, accuracy, temperature).
`bench/models/` contain the exact artifacts measured — rerun `evaluate.py`
on any x86 machine with the two runtimes installed to confirm accuracy;
latency numbers reproduce on a Pi 5 under the `performance` governor
(1-thread rows within ~2% noise).
