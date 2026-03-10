"""
save_train_scaled.py
====================
Generates panel_train_scaled.parquet by applying the fitted RobustScalers
to the raw training panel.

This file should be treated as a permanent pipeline artifact alongside
panel_val_scaled.parquet and panel_test_scaled.parquet.

Run this once on your Mac after Phase 3 + Phase 4 Step 1 are complete
(i.e. after fit_scalers.py has been run and config/scalers/ exists).

USAGE
-----
    cd /Users/ishraklatif/Documents/financial_data/Financial-data
    python -m scripts.build.save_train_scaled

OUTPUT
------
    data/features/panel_train_scaled.parquet  (~66 MB)

THEN
----
    Upload this file to Colab at:
    /content/drive/MyDrive/stockpred/repo/data/features/panel_train_scaled.parquet
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH   = PROJECT_ROOT / "data" / "features" / "panel_train.parquet"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "features" / "panel_train_scaled.parquet"
LOCKED_PATH  = PROJECT_ROOT / "config" / "features_locked.json"
SCALERS_DIR  = PROJECT_ROOT / "config" / "scalers"


def main():
    log.info("=" * 60)
    log.info("save_train_scaled.py")
    log.info("=" * 60)

    # ── Load locked features ──────────────────────────────────────────────
    log.info(f"Loading locked features from {LOCKED_PATH} ...")
    with open(LOCKED_PATH) as f:
        locked = json.load(f)
    features = locked["features"]
    log.info(f"  Locked features: {len(features)}")

    # ── Load raw train parquet ────────────────────────────────────────────
    log.info(f"Loading {TRAIN_PATH} ...")
    train = pd.read_parquet(TRAIN_PATH)
    log.info(f"  Shape: {train.shape}")

    # ── Verify scalers directory ──────────────────────────────────────────
    if not SCALERS_DIR.exists():
        log.error(f"Scalers directory not found: {SCALERS_DIR}")
        log.error("Run fit_scalers.py first (Phase 4 Step 1).")
        sys.exit(1)

    pkl_files = list(SCALERS_DIR.glob("scaler_*.pkl"))
    log.info(f"  Found {len(pkl_files)} scaler files in {SCALERS_DIR}")

    # ── Apply scalers ─────────────────────────────────────────────────────
    log.info("Applying scalers ...")
    n_scaled      = 0
    n_passthrough = 0
    n_missing     = 0

    for feat in features:
        path = SCALERS_DIR / f"scaler_{feat}.pkl"

        if not path.exists():
            # Passthrough feature (calendar indicators etc.) — no scaler file
            n_passthrough += 1
            continue

        if feat not in train.columns:
            log.warning(f"  Feature not in train parquet: {feat}")
            n_missing += 1
            continue

        with open(path, "rb") as f:
            scaler = pickle.load(f)

        col      = train[feat].astype("float64")
        not_null = col.notna()

        if not_null.any():
            col.loc[not_null] = scaler.transform(
                col.loc[not_null].values.reshape(-1, 1)
            ).flatten()

        train[feat] = col
        n_scaled += 1

        if n_scaled % 50 == 0:
            log.info(f"  ... {n_scaled} features scaled")

    log.info(f"  Scaled:      {n_scaled} features")
    log.info(f"  Passthrough: {n_passthrough} features (no scaler — expected)")
    log.info(f"  Missing:     {n_missing} features (check if dropped in lock step)")

    # ── Verify scaled values ──────────────────────────────────────────────
    log.info("Verifying scaled output ...")
    check = ["eq_rvol_63d", "F3_A_YIELD_3Y_lag", "eq_mom_252d", "cr_hy_spread"]
    for feat in check:
        if feat not in train.columns:
            continue
        col = train[feat].dropna()
        log.info(
            f"  {feat:<35}  "
            f"median={col.median():+.3f}  "
            f"IQR={col.quantile(0.75)-col.quantile(0.25):.3f}  "
            f"(should be median≈0, IQR≈1)"
        )

    # ── Save ──────────────────────────────────────────────────────────────
    keep_cols = ["date", "ticker", "y_ret_21d", "y_dir_21d"] + features
    keep_cols = [c for c in keep_cols if c in train.columns]
    train = train[keep_cols]

    log.info(f"Saving to {OUTPUT_PATH} ...")
    log.info(f"  Output shape: {train.shape}")
    train.to_parquet(OUTPUT_PATH, index=False, compression="snappy")


    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    log.info(f"  Saved: {OUTPUT_PATH.name}  ({size_mb:.1f} MB)")

    log.info("=" * 60)
    log.info("DONE — next steps:")
    log.info("  1. Upload panel_train_scaled.parquet to Colab at:")
    log.info("     /content/drive/MyDrive/stockpred/repo/data/features/")
    log.info("  2. Verify train_config.yaml points to panel_train_scaled.parquet")
    log.info("  3. Run training scripts in Colab")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
