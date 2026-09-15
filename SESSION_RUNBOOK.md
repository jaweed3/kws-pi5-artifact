# SESSION RUNBOOK — Pi5 revision experiments (ICAITech minor revision)

Target: 1 sesi pinjem, ~2-3 jam total. Urutan = payoff tertinggi dulu,
supaya kalau waktu kepotong, yang paling penting sudah dapat.

Prasyarat di Pi5 (repo sudah di-pull, branch `revise-icaitech-minor`):
  cd ~/kws-pi5-benchmark   # atau path repo di Pi5
  git pull && git checkout revise-icaitech-minor
  governor performance (script melakukannya otomatis via sudo tee)

Bawaan dari Mac (file kecil, via git pull juga bisa):
  bench/code/make_testlists.py bench/code/bench_threads.py
  bench/code/quantize_dscnn.py bench/code/make_rep_dscnn.py
  data/dscnn_rep.npz  (~2MB, cek: dibuat di Mac dari SCv2 train split)
  data/testlist_*.tsv TIDAK ikut git (bench/data/ di-gitignore) -> buat di Pi5

Dataset di Pi5: ../data/scv2_full (5.6GB, sudah ada dari eksperimen awal).
Jika direktori data berbeda, sesuaikan --data_dir di tiap perintah.

## 0. Bangun testlist (5 menit, sekali saja)

  cd bench/code
  python make_testlists.py --data_dir ../data/scv2_full --out_dir ../data
  # -> ../data/testlist_10.tsv (expect ~4,074 rows)
  # -> ../data/testlist_12.tsv (expect ~11,005 rows: 4,074 cmd + ~6k unknown + 1,000 silence)
  wc -l ../data/testlist_10.tsv ../data/testlist_12.tsv
  # VALIDASI: 10-list harus reproduksi angka paper (90.55/79.63/90.48/92.56)

## 1. Eval 12-class (HIGHEST PAYOFF, ~1 jam, jawab R1 point 4)

  # 4 model x 3 kolom (fp32 tflite, fp32 onnx, int8 tflite) x 2 list (10+12):
  for m in dnn lstm crnn; do
    python evaluate.py --model ../models/trained/$m/${m}_fp32.tflite --backend tflite --wavlist ../data/testlist_12.tsv
    python evaluate.py --model ../models/trained/$m/${m}_fp32.onnx   --backend onnx   --wavlist ../data/testlist_12.tsv
    python evaluate.py --model ../models/trained/$m/${m}_int8.tflite --backend tflite --wavlist ../data/testlist_12.tsv
  done
  # DS-CNN fp32 (rebuilt) + mixed:
  python evaluate.py --model ../models/kws_ref_model_fp32.tflite   --backend tflite --wavlist ../data/testlist_12.tsv
  python evaluate.py --model ../models/kws_ref_model_fp32.onnx     --backend onnx   --wavlist ../data/testlist_12.tsv
  python evaluate.py --model ../models/kws_ref_model_float32.tflite --backend tflite --wavlist ../data/testlist_12.tsv
  # DS-CNN full-int8 (setelah langkah 3):
  python evaluate.py --model ../models/kws_ref_model_fullint8.tflite --backend tflite --wavlist ../data/testlist_12.tsv
  python evaluate.py --model ../models/kws_ref_model_fullint8.tflite --backend tflite --wavlist ../data/testlist_10.tsv
  # SIMPAN semua stdout ke file: script tidak menulis JSON, jadi:
  #   ... | tee results/eval12_<model>_<backend>.txt

## 2. Thread scaling (MEDIUM, ~30 menit, jawab R1 point 3)

  python bench_threads.py --models dscnn,crnn --threads 1,2,4 --output results/bench_threads.json
  # VALIDASI: baris T=1 harus match benchmark_kws_pi.json (0.2107) & bench_crnn_fp32.json (0.2586)
  # dalam noise (<2%). Kalau TFLite num_threads diabaikan LiteRT, JSON mencatatnya jujur.

## 3. Full-int8 DS-CNN (MEDIUM, ~30 menit, jawab R1+R3 labeling)

  python quantize_dscnn.py --rep ../data/dscnn_rep.npz --out ../models/kws_ref_model_fullint8.tflite
  python benchmark_kws.py --tflite ../models/kws_ref_model_fullint8.tflite \
      --onnx ../models/kws_ref_model_fp32.onnx --tag dscnn_fullint8 \
      --wavlist ../data/testlist_10.tsv --output results/bench_dscnn_fullint8.json
  # (kolom onnx = referensi fp32; yang baru = kolom tflite)

## 4. Power/energy (LOW effort tambahan, bareng langkah 2, jawab R1 point 2)

  # USB meter inline ke Pi5. Catat watt idle, lalu watt tiap config saat run.
  # joules/inference = watt * mean_latency_s. Tabel: model x runtime -> W, mJ/inf.
  # Kalau meter tidak ada: SKIP, tulis future-work + estimasi literatur di paper.

## 5. Pulang: file yang dibawa pulang (git add + push dari Pi5)

  results/bench_threads.json results/bench_dscnn_fullint8.json
  results/eval12_*.txt  ../models/kws_ref_model_fullint8.tflite (26KB, boleh commit)
  power_notes.txt (catatan watt manual)
