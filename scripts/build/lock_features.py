#!/usr/bin/env python3
"""
lock_features.py
================
Phase 3, Step 4 — Feature selection and schema locking.

Reads panel_train.parquet and produces a locked feature list based on
two filters applied to the TRAIN split only:

  1. MISSING FILTER : drop any feature with >MISS_THRESHOLD null rate
                      in train. Default: 50%.
                      Removes late-start series (CSI300, XLRE, XLC)
                      and any series with chronic data gaps.

  2. VARIANCE FILTER: drop any feature with variance < VAR_THRESHOLD
                      in train. Default: 1e-8.
                      Removes constant or near-constant columns that
                      carry no predictive information.

The surviving feature list is written to config/features_locked.json
along with a SHA-256 schema hash. All downstream scripts (train_model,
assert_schema) read this file — it is the single source of truth for
the feature set.

LEAKAGE SAFETY
--------------
Filters are computed ONLY on train. Val and test are never inspected
during locking. This ensures that the decision of which features to
include cannot be influenced by knowledge of val/test distributions.

OUTPUTS
-------
config/features_locked.json
    {
      "features": [...],          # ordered list of surviving column names
      "count": N,
      "schema_hash": "abc123...", # SHA-256 of the feature list
      "generated_at": "...",
      "filters": {
        "miss_threshold": 0.50,
        "var_threshold": 1e-8,
        "train_rows": N,
        "dropped_missing": [...],
        "dropped_variance": [...],
        "surviving": N
      }
    }

data/audit/lock_features_report.json
    Full per-column stats: null_rate, variance, kept/dropped, reason.

USAGE
-----
    python -m scripts.build.lock_features
    python -m scripts.build.lock_features --verify
    python -m scripts.build.lock_features --report   # print dropped cols
    python -m scripts.build.lock_features --threshold 0.4  # custom miss threshold
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "lock_features.log"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
CONFIG_DIR   = PROJECT_ROOT / "config"
AUDIT_DIR    = PROJECT_ROOT / "data" / "audit"

TRAIN_PATH   = FEATURES_DIR / "panel_train.parquet"
LOCKED_PATH  = CONFIG_DIR   / "features_locked.json"
REPORT_PATH  = AUDIT_DIR    / "lock_features_report.json"

# ─────────────────────────────────────────────────────────────────────────────
# Filter thresholds
# ─────────────────────────────────────────────────────────────────────────────

MISS_THRESHOLD = 0.50   # drop features with >50% null in train
VAR_THRESHOLD  = 1e-8   # drop features with variance below this

# Columns that are never features — always excluded
NON_FEATURE_COLS = {"date", "ticker", "y_ret_21d", "y_dir_21d"}


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_column_stats(train: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Compute null rate and variance for each candidate column on train."""
    stats = []
    for col in candidates:
        series = train[col]
        null_rate = float(series.isna().mean())
        variance  = float(series.var()) if series.notna().sum() > 1 else 0.0
        stats.append({
            "column":    col,
            "null_rate": round(null_rate, 6),
            "variance":  variance,
            "n_nonull":  int(series.notna().sum()),
        })
    return pd.DataFrame(stats).set_index("column")


def apply_filters(
    stats: pd.DataFrame,
    miss_threshold: float,
    var_threshold: float,
) -> tuple[list[str], list[str], list[str]]:
    """
    Apply missing and variance filters.
    Returns (kept, dropped_missing, dropped_variance).
    """
    dropped_missing  = stats[stats["null_rate"] > miss_threshold].index.tolist()
    remaining        = stats[stats["null_rate"] <= miss_threshold]
    dropped_variance = remaining[remaining["variance"] < var_threshold].index.tolist()
    kept             = remaining[remaining["variance"] >= var_threshold].index.tolist()
    return sorted(kept), sorted(dropped_missing), sorted(dropped_variance)


def schema_hash(features: list[str]) -> str:
    """SHA-256 of the sorted, joined feature list."""
    return hashlib.sha256(json.dumps(features).encode()).hexdigest()


def build_report(
    stats: pd.DataFrame,
    kept: list[str],
    dropped_missing: list[str],
    dropped_variance: list[str],
    miss_threshold: float,
    var_threshold: float,
) -> dict:
    """Build full per-column audit report."""
    records = []
    for col in stats.index:
        if col in dropped_missing:
            status = "DROPPED_MISSING"
            reason = f"null_rate={stats.loc[col,'null_rate']:.1%} > {miss_threshold:.0%}"
        elif col in dropped_variance:
            status = "DROPPED_VARIANCE"
            reason = f"variance={stats.loc[col,'variance']:.2e} < {var_threshold:.0e}"
        else:
            status = "KEPT"
            reason = ""
        records.append({
            "column":    col,
            "status":    status,
            "null_rate": stats.loc[col, "null_rate"],
            "variance":  stats.loc[col, "variance"],
            "n_nonull":  int(stats.loc[col, "n_nonull"]),
            "reason":    reason,
        })

    return {
        "generated_at":      datetime.utcnow().isoformat(),
        "miss_threshold":    miss_threshold,
        "var_threshold":     var_threshold,
        "total_candidates":  len(stats),
        "kept":              len(kept),
        "dropped_missing":   len(dropped_missing),
        "dropped_variance":  len(dropped_variance),
        "columns":           records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(miss_threshold: float = MISS_THRESHOLD,
        var_threshold:  float = VAR_THRESHOLD,
        print_report:   bool  = False) -> None:

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("lock_features.py  Phase 3 Step 4")
    log.info(f"Train   : {TRAIN_PATH}")
    log.info(f"Filters : miss>{miss_threshold:.0%}  var<{var_threshold:.0e}")
    log.info("=" * 65)

    if not TRAIN_PATH.exists():
        log.error(f"panel_train.parquet not found. Run split_and_fill.py first.")
        sys.exit(1)

    # ── Load train only ───────────────────────────────────────────────────
    log.info("Loading panel_train.parquet (train split only) ...")
    train = pd.read_parquet(TRAIN_PATH)
    log.info(f"  Rows: {len(train):,}  |  Tickers: {train['ticker'].nunique()}  |  "
             f"Cols: {len(train.columns)}")

    # ── Identify candidate features ───────────────────────────────────────
    candidates = [c for c in train.columns if c not in NON_FEATURE_COLS]
    log.info(f"  Candidate features: {len(candidates)}")

    # ── Compute stats ─────────────────────────────────────────────────────
    log.info("Computing null rates and variance on train ...")
    stats = compute_column_stats(train, candidates)

    # ── Apply filters ─────────────────────────────────────────────────────
    log.info("Applying filters ...")
    kept, dropped_missing, dropped_variance = apply_filters(
        stats, miss_threshold, var_threshold
    )

    log.info(f"  Candidates      : {len(candidates)}")
    log.info(f"  Dropped missing : {len(dropped_missing)}")
    log.info(f"  Dropped variance: {len(dropped_variance)}")
    log.info(f"  Kept            : {len(kept)}")

    # ── Log dropped columns ───────────────────────────────────────────────
    if dropped_missing:
        log.info(f"  Dropped (missing >{ miss_threshold:.0%}):")
        for col in dropped_missing:
            log.info(f"    {col:<50} null={stats.loc[col,'null_rate']:.1%}")

    if dropped_variance:
        log.info(f"  Dropped (variance <{var_threshold:.0e}):")
        for col in dropped_variance:
            log.info(f"    {col:<50} var={stats.loc[col,'variance']:.2e}")

    if print_report:
        log.info(f"  Kept features ({len(kept)}):")
        for col in kept:
            log.info(f"    {col:<50} null={stats.loc[col,'null_rate']:.1%}  "
                     f"var={stats.loc[col,'variance']:.4f}")

    # ── Build locked config ───────────────────────────────────────────────
    h = schema_hash(kept)
    locked = {
        "generated_at":  datetime.utcnow().isoformat(),
        "schema_hash":   h,
        "count":         len(kept),
        "features":      kept,
        "filters": {
            "miss_threshold":    miss_threshold,
            "var_threshold":     var_threshold,
            "train_rows":        int(len(train)),
            "train_tickers":     int(train["ticker"].nunique()),
            "dropped_missing":   dropped_missing,
            "dropped_variance":  dropped_variance,
        },
    }

    # ── Save ──────────────────────────────────────────────────────────────
    with open(LOCKED_PATH, "w") as f:
        json.dump(locked, f, indent=2)
    log.info(f"  Locked config   : {LOCKED_PATH}")
    log.info(f"  Schema hash     : {h[:16]}...")

    report = build_report(
        stats, kept, dropped_missing, dropped_variance,
        miss_threshold, var_threshold
    )
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"  Audit report    : {REPORT_PATH}")

    # ── Verify on val and test ────────────────────────────────────────────
    log.info("Verifying locked features exist in val and test ...")
    for split_name in ["val", "test"]:
        split_path = FEATURES_DIR / f"panel_{split_name}.parquet"
        if not split_path.exists():
            log.warning(f"  {split_name}: not found, skipping check")
            continue
        df = pd.read_parquet(split_path)
        missing_in_split = set(kept) - set(df.columns)
        if missing_in_split:
            log.error(f"  {split_name}: {len(missing_in_split)} locked features missing!")
            for c in sorted(missing_in_split):
                log.error(f"    {c}")
        else:
            log.info(f"  {split_name}: all {len(kept)} features present ✓")

    log.info("=" * 65)
    log.info("LOCK FEATURES COMPLETE")
    log.info(f"  Locked features : {len(kept)}")
    log.info(f"  Schema hash     : {h[:16]}...")
    log.info(f"  Config          : {LOCKED_PATH}")
    log.info("=" * 65)


def verify() -> None:
    log.info("=" * 65)
    log.info("VERIFICATION — features_locked.json")

    if not LOCKED_PATH.exists():
        log.error(f"features_locked.json not found at {LOCKED_PATH}")
        sys.exit(1)

    with open(LOCKED_PATH) as f:
        locked = json.load(f)

    features = locked["features"]
    h = schema_hash(features)

    log.info(f"  Feature count   : {locked['count']}")
    log.info(f"  Schema hash     : {locked['schema_hash'][:16]}...")
    log.info(f"  Hash verified   : {'✓ OK' if h == locked['schema_hash'] else '✗ MISMATCH'}")
    log.info(f"  Generated at    : {locked['generated_at']}")
    log.info(f"  Miss threshold  : {locked['filters']['miss_threshold']:.0%}")
    log.info(f"  Train rows      : {locked['filters']['train_rows']:,}")
    log.info(f"  Dropped missing : {len(locked['filters']['dropped_missing'])}")
    log.info(f"  Dropped variance: {len(locked['filters']['dropped_variance'])}")

    if h != locked["schema_hash"]:
        log.error("  Schema hash mismatch — features_locked.json is corrupt")
        sys.exit(1)

    # Check all splits contain the locked features
    for split_name in ["train", "val", "test"]:
        path = FEATURES_DIR / f"panel_{split_name}.parquet"
        if not path.exists():
            log.warning(f"  {split_name}: not found")
            continue
        df = pd.read_parquet(path)
        missing = set(features) - set(df.columns)
        extra   = set(df.columns) - set(features) - NON_FEATURE_COLS
        log.info(f"  {split_name:<6}: locked cols present={len(features)-len(missing)}/{len(features)}  "
                 f"extra_cols={len(extra)}")
        if missing:
            log.error(f"    Missing: {sorted(missing)[:5]}")

    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",    action="store_true",
                        help="Verify existing features_locked.json")
    parser.add_argument("--report",    action="store_true",
                        help="Print all kept features with stats")
    parser.add_argument("--threshold", type=float, default=MISS_THRESHOLD,
                        help=f"Missing rate threshold (default: {MISS_THRESHOLD})")
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        run(
            miss_threshold=args.threshold,
            var_threshold=VAR_THRESHOLD,
            print_report=args.report,
        )