"""
colab_setup.py
==============
Google Colab setup script for Phase 4 training.

Run this FIRST in a new Colab session before any training script.
It handles:
  1. Drive mounting
  2. Repository clone / sync from Drive
  3. Dependency installation
  4. GPU verification
  5. Data file verification

COLAB USAGE (paste into a cell):
---------------------------------
    !wget -q https://... colab_setup.py  # or upload manually
    !python colab_setup.py --drive-dir /content/drive/MyDrive/stockpred

OR paste each section as separate Colab cells (recommended).

RECOMMENDED COLAB CELL STRUCTURE
---------------------------------
Cell 1 — Mount Drive:
    from google.colab import drive
    drive.mount('/content/drive')

Cell 2 — Install deps:
    !pip install -q lightgbm pytorch-lightning pyyaml scipy scikit-learn

Cell 3 — Clone or sync repo:
    import shutil, os
    DRIVE_DIR = '/content/drive/MyDrive/stockpred'
    REPO_DIR  = '/content/stockpred'
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    shutil.copytree(DRIVE_DIR + '/repo', REPO_DIR)
    os.chdir(REPO_DIR)

Cell 4 — Verify GPU:
    import torch
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"GPU:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

Cell 5 — Verify data:
    import os
    files = [
        'data/features/panel_train.parquet',
        'data/features/panel_val_scaled.parquet',
        'data/features/panel_test_scaled.parquet',
        'config/features_locked.json',
        'config/train_config.yaml',
    ]
    for f in files:
        size = os.path.getsize(f) / 1e6
        print(f"{'OK' if os.path.exists(f) else 'MISSING':6} {f}  ({size:.1f} MB)")

Cell 6a — Train LightGBM (fastest, run first):
    !python -m scripts.train.train_lightgbm \\
        --drive-dir /content/drive/MyDrive/stockpred/checkpoints

Cell 6b — Train DLinear:
    !python -m scripts.train.train_dlinear \\
        --drive-dir /content/drive/MyDrive/stockpred/checkpoints

Cell 6c — Train FT-Transformer:
    !python -m scripts.train.train_ft_transformer \\
        --drive-dir /content/drive/MyDrive/stockpred/checkpoints

Cell 6d — Train TFT (most memory-intensive, run last):
    !python -m scripts.train.train_tft \\
        --drive-dir /content/drive/MyDrive/stockpred/checkpoints

Cell 7 — Resume after disconnect:
    # Each model auto-resumes from the last checkpoint saved to Drive
    !python -m scripts.train.train_tft \\
        --drive-dir /content/drive/MyDrive/stockpred/checkpoints \\
        --resume

DRIVE DIRECTORY STRUCTURE
--------------------------
/content/drive/MyDrive/stockpred/
    repo/                    ← copy your entire project here
        config/
        data/
        scripts/
        ...
    checkpoints/             ← training checkpoints (auto-created)
        lightgbm/
            lgbm_reg.txt
            lgbm_cls.txt
            results.json
        dlinear/
            best.pt
            latest.pt
        ft_transformer/
            best.pt
            latest.pt
        tft/
            best.pt
            latest.pt

MEMORY REQUIREMENTS (A100 40GB)
---------------------------------
LightGBM     : ~2 GB RAM (CPU-only), no GPU memory
DLinear      : ~4 GB VRAM  (batch=512, lookback=126, features=260)
FT-Transformer: ~6 GB VRAM  (batch=1024, d_token=192, n_blocks=3)
TFT          : ~12 GB VRAM (batch=256, d_model=128, lookback=126)

If you get OOM on TFT, reduce batch_size in train_config.yaml:
    tft:
      batch_size: 128   # reduce from 256

TRAINING TIME ESTIMATES (A100)
--------------------------------
LightGBM     : 5–15 minutes
DLinear      : 1–2 hours (150 epochs)
FT-Transformer: 2–4 hours (100 epochs)
TFT          : 4–8 hours (100 epochs)

RECOMMENDED ORDER
-----------------
1. LightGBM first — establishes the baseline IC in ~10 minutes
2. DLinear second — fastest deep model, validates the sequence pipeline
3. FT-Transformer third — validates the tabular transformer pipeline
4. TFT last — most complex, most memory, most likely to need hyperparameter tuning

WHAT GOOD RESULTS LOOK LIKE
----------------------------
For an ASX 50 universe with 21-day return prediction:
  IC > 0.03   — signal present (statistically)
  IC > 0.06   — commercially useful signal
  IC > 0.10   — strong signal (rare for a single model on real data)
  ICIR > 0.5  — consistent signal (IC / std(IC))
  L/S Sharpe > 1.0 — production-viable after costs at this universe size
  Hit Rate > 53% — directional accuracy above chance

If LightGBM val IC < 0.02, check:
  - Are scalers applied correctly? (check panel_val_scaled.parquet)
  - Is the target correct? (check y_ret_21d distribution on val)
  - Are features actually predictive? (check top IC features from audit)

TROUBLESHOOTING
---------------
OOM on TFT:       reduce batch_size to 128 in train_config.yaml
Slow DLinear:     set num_workers=0 in build_loaders call
NaN loss:         reduce learning_rate by 10×; check for NaN in features
Poor IC on all:   verify scalers were applied (panel_val_scaled, not panel_val)
Session disconnect: re-run setup cells 1-5, then run --resume
"""

# This file is documentation only — no executable code.
# All training is done via the individual train_*.py scripts.
print(__doc__)
