#!/usr/bin/env python3
"""
fit_scalers.py
==============
Phase 4, Step 1 — Fit feature scalers on train split only.

Reads panel_train.parquet and config/features_locked.json, then fits
one RobustScaler per feature using train data only. Scalers are saved
to config/scalers/ as individual .pkl files plus a combined manifest.

WHY ROBUSTSCALER
----------------
RobustScaler uses the median and IQR (interquartile range) rather than
mean and std. This makes it resilient to the fat tails and outlier
returns inherent in financial time series. For example:
  - eq_mom_252d (12-month momentum) has occasional extreme values
    during GFC / COVID that would distort a StandardScaler
  - xa_iron_close has a long right tail from commodity supercycles
  - si_short_pct has sparse, skewed distributions

WHY FIT ON TRAIN ONLY
---------------------
Fitting scalers on the full dataset (train + val + test) is a form of
data leakage: the scaler would encode distributional information from
future periods into the transformation applied to past periods. The
scaler must be fit exclusively on train and then APPLIED (transform
only, never fit_transform) to val and test.

NULL HANDLING
-------------
NaN values are excluded from scaler fitting (sklearn handles this via
masking). During transform, NaN values are preserved as NaN — the model
must handle them separately (via masking, imputation, or embedding).

BINARY / LOW-CARDINALITY FEATURES
----------------------------------
Calendar indicator features (cal_is_month_end, cal_is_quarter_end,
cal_is_year_end, cal_day_of_week) are flagged as passthrough — they
are not scaled because:
  1. They are already in a bounded [0,1] or [0,4] range
  2. Scaling them distorts their categorical meaning
  3. They have very low variance; IQR-based scaling can produce
     numerically unstable transforms (divide by near-zero IQR)

OUTPUTS
-------
config/scalers/
    scaler_<feature_name>.pkl   — individual RobustScaler per feature
    scalers_manifest.json       — metadata: feature list, fit stats,
                                  passthrough list, schema hash

USAGE
-----
    python -m scripts.build.fit_scalers
    python -m scripts.build.fit_scalers --verify        # check saved scalers
    python -m scripts.build.fit_scalers --report        # print fit statistics
    python -m scripts.build.fit_scalers --apply val     # apply to val, save scaled
    python -m scripts.build.fit_scalers --apply test    # apply to test, save scaled
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "fit_scalers.log"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
CONFIG_DIR    = PROJECT_ROOT / "config"
SCALERS_DIR   = CONFIG_DIR / "scalers"
LOCKED_PATH   = CONFIG_DIR / "features_locked.json"
MANIFEST_PATH = SCALERS_DIR / "scalers_manifest.json"

SPLIT_PATHS = {
    "train": FEATURES_DIR / "panel_train.parquet",
    "val":   FEATURES_DIR / "panel_val.parquet",
    "test":  FEATURES_DIR / "panel_test.parquet",
}

SCALED_PATHS = {
    "val":  FEATURES_DIR / "panel_val_scaled.parquet",
    "test": FEATURES_DIR / "panel_test_scaled.parquet",
}

# ─────────────────────────────────────────────────────────────────────────────
# Passthrough features (not scaled)
# ─────────────────────────────────────────────────────────────────────────────

# These features are passed through unchanged. Either they are binary
# indicators, low-cardinality categoricals, or already in a natural
# bounded range where scaling is harmful.
PASSTHROUGH_PREFIXES = (
    "cal_is_",        # binary indicators: month_end, quarter_end, year_end
    "cal_day_of_week", # ordinal 0-4, categorical meaning
    "cal_month",       # ordinal 1-12, cyclical — handle separately if needed
    "cal_quarter",     # ordinal 1-4
    "cal_week_of_year",# ordinal 1-53
    "cal_year",        # year integer — keep as-is for trend features
    "td_idx",          # global trading day index — monotonic, used as positional
)


def is_passthrough(col: str) -> bool:
    return any(col.startswith(pfx) for pfx in PASSTHROUGH_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def load_locked() -> dict:
    if not LOCKED_PATH.exists():
        log.error(f"features_locked.json not found: {LOCKED_PATH}")
        sys.exit(1)
    with open(LOCKED_PATH) as f:
        return json.load(f)


def fit_scaler(series: pd.Series) -> RobustScaler:
    """
    Fit a RobustScaler on a single feature series, ignoring NaN values.
    Returns the fitted scaler.
    """
    values = series.dropna().values.reshape(-1, 1)
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(values)
    return scaler


def scaler_stats(scaler: RobustScaler, series: pd.Series) -> dict:
    """Extract human-readable stats from a fitted scaler."""
    return {
        "center":     float(scaler.center_[0]),
        "scale":      float(scaler.scale_[0]),
        "n_fit":      int(series.notna().sum()),
        "null_pct":   round(float(series.isna().mean() * 100), 2),
        "raw_min":    round(float(series.min()), 6),
        "raw_max":    round(float(series.max()), 6),
        "raw_median": round(float(series.median()), 6),
        "raw_p25":    round(float(series.quantile(0.25)), 6),
        "raw_p75":    round(float(series.quantile(0.75)), 6),
    }


def apply_scaler(
    df: pd.DataFrame,
    feature: str,
    scaler: RobustScaler,
) -> pd.Series:
    """
    Apply a fitted scaler to a column, preserving NaN values.
    Returns a new float64 Series with scaled values.

    Always returns float64 regardless of the source dtype. This avoids
    a pandas FutureWarning when writing float values into integer columns
    (e.g. cal_dist_to_month_end stored as int64 in train but scaled here).
    """
    col = df[feature].copy().astype("float64")  # upcast first — avoids FutureWarning
    not_null = col.notna()
    if not_null.any():
        col.loc[not_null] = scaler.transform(
            col.loc[not_null].values.reshape(-1, 1)
        ).flatten()
    return col


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(report: bool = False) -> None:
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("fit_scalers.py  Phase 4 Step 1")
    log.info(f"Train   : {SPLIT_PATHS['train']}")
    log.info(f"Scalers : {SCALERS_DIR}")
    log.info("=" * 65)

    # ── Load locked features ──────────────────────────────────────────────
    locked  = load_locked()
    features = locked["features"]
    schema_hash = locked["schema_hash"]
    log.info(f"Locked features : {len(features)}  |  Hash: {schema_hash[:16]}...")

    # ── Identify passthrough features ─────────────────────────────────────
    passthrough = [f for f in features if is_passthrough(f)]
    to_scale    = [f for f in features if not is_passthrough(f)]
    log.info(f"To scale        : {len(to_scale)}")
    log.info(f"Passthrough     : {len(passthrough)}  ({', '.join(passthrough[:4])} ...)")

    # ── Load train split ──────────────────────────────────────────────────
    log.info("Loading panel_train.parquet ...")
    train = pd.read_parquet(SPLIT_PATHS["train"])
    log.info(f"  Rows: {len(train):,}  |  Tickers: {train['ticker'].nunique()}")

    # Verify all features present
    missing = [f for f in features if f not in train.columns]
    if missing:
        log.error(f"Missing features in train: {missing[:5]}")
        sys.exit(1)

    # ── Fit scalers ───────────────────────────────────────────────────────
    log.info(f"Fitting {len(to_scale)} scalers on train (RobustScaler, IQR) ...")

    scalers = {}
    stats   = {}
    failed  = []

    for i, feat in enumerate(to_scale):
        series = train[feat]
        n_valid = series.notna().sum()

        if n_valid < 100:
            log.warning(f"  SKIP {feat}: only {n_valid} non-null values in train")
            failed.append(feat)
            continue

        scaler = fit_scaler(series)

        # Guard: if IQR is near zero, scaler is numerically unstable
        if scaler.scale_[0] < 1e-10:
            log.warning(f"  WARN {feat}: near-zero IQR ({scaler.scale_[0]:.2e}) "
                        f"— treating as passthrough")
            passthrough.append(feat)
            to_scale.remove(feat)
            continue

        scalers[feat] = scaler
        stats[feat]   = scaler_stats(scaler, series)

        if report:
            s = stats[feat]
            log.info(f"  {feat:<45} "
                     f"center={s['center']:>10.4f}  "
                     f"scale={s['scale']:>10.4f}  "
                     f"null={s['null_pct']:.1f}%")

        if (i + 1) % 50 == 0:
            log.info(f"  ... {i+1}/{len(to_scale)} scalers fit")

    log.info(f"  Fit complete: {len(scalers)} scalers  |  "
             f"{len(failed)} skipped  |  "
             f"{len(passthrough)} passthrough")

    # ── Save scalers ──────────────────────────────────────────────────────
    log.info("Saving scalers ...")
    for feat, scaler in scalers.items():
        path = SCALERS_DIR / f"scaler_{feat}.pkl"
        with open(path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── Save manifest ─────────────────────────────────────────────────────
    manifest = {
        "generated_at":   datetime.utcnow().isoformat(),
        "schema_hash":    schema_hash,
        "scaler_type":    "RobustScaler",
        "quantile_range": [25.0, 75.0],
        "train_rows":     int(len(train)),
        "n_scaled":       len(scalers),
        "n_passthrough":  len(passthrough),
        "n_skipped":      len(failed),
        "scaled_features":     sorted(scalers.keys()),
        "passthrough_features": sorted(passthrough),
        "skipped_features":    sorted(failed),
        "fit_stats":      stats,
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("=" * 65)
    log.info("FIT SCALERS COMPLETE")
    log.info(f"  Scalers saved   : {len(scalers)} files in {SCALERS_DIR}")
    log.info(f"  Manifest        : {MANIFEST_PATH}")
    log.info(f"  Passthrough     : {len(passthrough)} features (not scaled)")
    log.info(f"  Skipped         : {len(failed)} features (too sparse)")
    log.info("=" * 65)


def verify() -> None:
    """Check that all expected scaler files exist and load cleanly."""
    if not MANIFEST_PATH.exists():
        log.error("scalers_manifest.json not found — run fit_scalers first")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    log.info("=" * 65)
    log.info("VERIFICATION — scalers")
    log.info(f"  Generated at    : {manifest['generated_at']}")
    log.info(f"  Schema hash     : {manifest['schema_hash'][:16]}...")
    log.info(f"  Scaler type     : {manifest['scaler_type']}")
    log.info(f"  Scaled features : {manifest['n_scaled']}")
    log.info(f"  Passthrough     : {manifest['n_passthrough']}")

    # Check each expected scaler file exists
    missing = []
    corrupt = []
    for feat in manifest["scaled_features"]:
        path = SCALERS_DIR / f"scaler_{feat}.pkl"
        if not path.exists():
            missing.append(feat)
            continue
        try:
            with open(path, "rb") as f:
                scaler = pickle.load(f)
            assert hasattr(scaler, "center_"), f"Bad scaler for {feat}"
        except Exception as e:
            corrupt.append((feat, str(e)))

    if missing:
        log.error(f"  MISSING scaler files: {len(missing)}")
        for f in missing[:5]:
            log.error(f"    {f}")
        sys.exit(1)
    else:
        log.info(f"  [OK  ] All {manifest['n_scaled']} scaler files present and loadable")

    if corrupt:
        log.error(f"  CORRUPT scalers: {len(corrupt)}")
        sys.exit(1)

    # Verify schema hash still matches features_locked.json
    if LOCKED_PATH.exists():
        with open(LOCKED_PATH) as f:
            locked = json.load(f)
        if locked["schema_hash"] != manifest["schema_hash"]:
            log.error("  Schema hash mismatch — features_locked.json has changed since scalers were fit")
            log.error("  Re-run fit_scalers.py to regenerate")
            sys.exit(1)
        log.info(f"  [OK  ] Schema hash matches features_locked.json")

    log.info("=" * 65)


def apply_to_split(split_name: str) -> None:
    """
    Apply fitted scalers to val or test split and save scaled parquet.

    The scaled parquet has the same schema as the input but with
    continuous feature columns transformed. Passthrough and NaN values
    are preserved unchanged.
    """
    if split_name not in ("val", "test"):
        log.error(f"--apply must be 'val' or 'test', got: {split_name}")
        sys.exit(1)

    if not MANIFEST_PATH.exists():
        log.error("scalers_manifest.json not found — run fit_scalers first")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    log.info("=" * 65)
    log.info(f"APPLYING SCALERS — {split_name.upper()}")

    # ── Load split ────────────────────────────────────────────────────────
    path = SPLIT_PATHS[split_name]
    if not path.exists():
        log.error(f"Split not found: {path}")
        sys.exit(1)

    log.info(f"Loading {split_name} ...")
    df = pd.read_parquet(path)
    log.info(f"  Rows: {len(df):,}  |  Tickers: {df['ticker'].nunique()}")

    # ── Load scalers ──────────────────────────────────────────────────────
    log.info(f"Loading {len(manifest['scaled_features'])} scalers ...")
    scalers = {}
    for feat in manifest["scaled_features"]:
        scaler_path = SCALERS_DIR / f"scaler_{feat}.pkl"
        if not scaler_path.exists():
            log.error(f"Scaler missing: {feat}")
            sys.exit(1)
        with open(scaler_path, "rb") as f:
            scalers[feat] = pickle.load(f)

    # ── Apply ─────────────────────────────────────────────────────────────
    log.info("Applying scalers ...")
    df_scaled = df.copy()

    for feat, scaler in scalers.items():
        if feat not in df_scaled.columns:
            log.warning(f"  Feature {feat} not in {split_name} — skipping")
            continue
        df_scaled[feat] = apply_scaler(df_scaled, feat, scaler)

    # ── Spot-check: scaled values should be centred near 0 ON TRAIN ──────
    # NOTE: Val/test medians will NOT be 0 for level features (CPI, yield
    # levels, etc.) because those features have shifted significantly from
    # their 2005-2019 train distribution. This is CORRECT behaviour — the
    # scaler encodes "how far has this feature moved from its train centre",
    # which is exactly the cross-regime signal the model needs. Only worry
    # if a feature that should be stationary (e.g. a z-score or ratio) shows
    # a large median offset.
    log.info("Spot-check scaled distribution (median≈0 expected on train; "
             "val/test offsets reflect regime shift — see PSI audit):")
    sample_feats = list(scalers.keys())[:5]
    for feat in sample_feats:
        vals = df_scaled[feat].dropna()
        log.info(f"  {feat:<40} "
                 f"median={vals.median():>+.4f}  "
                 f"p25={vals.quantile(0.25):>+.4f}  "
                 f"p75={vals.quantile(0.75):>+.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = SCALED_PATHS[split_name]
    df_scaled.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    size_mb = out_path.stat().st_size / 1e6
    log.info(f"Saved: {out_path.name}  ({size_mb:.1f} MB)")
    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Check saved scalers without refitting")
    parser.add_argument("--report", action="store_true",
                        help="Print per-feature fit statistics during fitting")
    parser.add_argument("--apply", metavar="SPLIT",
                        help="Apply fitted scalers to val or test split")
    args = parser.parse_args()

    if args.verify:
        verify()
    elif args.apply:
        apply_to_split(args.apply)
    else:
        run(report=args.report)